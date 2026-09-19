#!/usr/bin/env python3
"""
AI Agent for Automated Food Image Collection & Processing.

Pipeline: Excel -> Understand Name -> Image Search -> Evaluate -> Process -> Rename -> Drive Upload
Usage:
    python agent.py --input input.xlsx --limit 5
    python agent.py --input input.xlsx --force
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import openpyxl
import requests
from PIL import Image
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_WIDTH = 1800
TARGET_HEIGHT = 1200
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT  # 1.5 (3:2)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024       # 10 MB
MIN_SOURCE_WIDTH = 900
MIN_SOURCE_HEIGHT = 600
BLUR_THRESHOLD = 100.0          # Laplacian variance; below = blurry
MAX_CANDIDATES = 6              # images to evaluate per item
MAX_SEARCH_RESULTS = 15         # results per search query
GEMINI_BATCH_SIZE = 40          # names per Gemini understanding call
JPEG_INITIAL_QUALITY = 90
DOWNLOAD_TIMEOUT = 15           # seconds
MAX_RAW_DOWNLOAD_BYTES = 15 * 1024 * 1024
SEARCH_DELAY_S = 1.5            # seconds between DuckDuckGo queries
GEMINI_DELAY_S = 0.5            # seconds between Gemini calls
DRIVE_RETRY_COUNT = 3
DRIVE_RETRY_BACKOFF_S = 2
UPSCALE_WARNING_FACTOR = 2.0
ILLEGAL_FILENAME_RE = re.compile(r'[\\/:*?"<>|]')
OUTPUT_DIR = Path("output/images")
TEMP_DIR = OUTPUT_DIR / "_temp"

# ═══════════════════════════════════════════════════════════════════════════════
# ASSUMPTIONS (also written to the report and README)
# ═══════════════════════════════════════════════════════════════════════════════
ASSUMPTIONS = [
    "Duplicate item names are processed once and reused for every row that has that name.",
    "A cell with two dishes separated by a line break is treated as two separate items.",
    "Variant names ([Half], (Half), Full, Special, etc.) are searched using the base dish, "
    "but the saved file keeps the exact name from Excel.",
    "Names that look like two dishes glued together with no line break are searched as written "
    "and may be flagged 'Needs Review'.",
    "Images come from public web search (DuckDuckGo) and are for internal restaurant-menu use; "
    "the source URL of every image is logged in the report. Minimum accepted source size is 900x600 px.",
    "If no image passes all checks, the best borderline image is used and flagged for manual review.",
    "Output format is JPG at 1800x1200 (3:2), quality ~90, always under 10 MB.",
    "The 'Google drive link' column of the input sheet is filled in output.xlsx "
    "(the original file is not modified).",
]

# Logging setup (file + console)
log = logging.getLogger("agent")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Food Image Automation Agent")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N unique items")
    parser.add_argument("--force", action="store_true", help="Re-process items already marked done")
    return parser.parse_args()


def setup_logging():
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    # File handler
    fh = logging.FileHandler("agent.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)


def setup_gemini():
    """Return (client, model_name) or (None, model_name) if unavailable."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
    if not api_key:
        log.warning("GEMINI_API_KEY not set. LLM features disabled (using regex fallback).")
        return None, model
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        log.info(f"Gemini ready  (model: {model})")
        return client, model
    except Exception as e:
        log.warning(f"Gemini init failed: {e}. Using regex fallback.")
        return None, model


def gemini_generate(client, model, contents, json_mode=False, max_retries=3):
    """Call Gemini with retries on rate-limit / transient errors."""
    from google.genai import types as gt
    config = gt.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            return resp.text
        except Exception as e:
            err = str(e).lower()
            retriable = any(k in err for k in ("429", "rate", "quota", "resource", "unavailable", "500", "503"))
            if attempt < max_retries - 1 and retriable:
                wait = (2 ** attempt) * 2
                log.warning(f"Gemini transient error, retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    return None


def parse_json_response(text):
    """Parse JSON from model output, stripping markdown fences if present."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object/array in text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            s = text.find(start_char)
            e = text.rfind(end_char)
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(text[s:e + 1])
                except json.JSONDecodeError:
                    continue
        return None


def sanitize_filename(name):
    """Replace only characters illegal in filenames; keep [ ] ( ) etc."""
    return ILLEGAL_FILENAME_RE.sub("_", name).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEL I/O
# ═══════════════════════════════════════════════════════════════════════════════

def read_excel(path):
    """
    Read the input Excel file.
    Returns:
        unique_items: list[str] – unique item names in original order
        item_rows:    dict[str, list[tuple[int, int]]] – name -> [(row_idx, col_idx)]
                      so we know where to write drive links back
    """
    wb = openpyxl.load_workbook(path)
    ws = wb.active  # fallback if sheet name differs
    for name in wb.sheetnames:
        if "table" in name.lower():
            ws = wb[name]
            break

    # Auto-detect header row (contains "item_name")
    header_row = None
    name_col = link_col = None
    for row in ws.iter_rows(min_row=1, max_row=10):
        for cell in row:
            val = str(cell.value).strip().lower() if cell.value else ""
            if val == "item_name":
                header_row = cell.row
                name_col = cell.column
            if "google" in val and "link" in val:
                link_col = cell.column
    if header_row is None or name_col is None:
        raise ValueError("Could not find 'item_name' header in the first 10 rows.")
    log.info(f"Header on row {header_row}, item_name col {name_col}, link col {link_col}")

    # Collect items
    seen = set()
    unique_items = []
    item_rows = {}  # name -> [(row, is_multiline_index)]
    for row in ws.iter_rows(min_row=header_row + 1):
        raw = row[name_col - 1].value
        if not raw:
            continue
        parts = [p.strip() for p in str(raw).split("\n") if p.strip()]
        for idx, name in enumerate(parts):
            if name not in seen:
                seen.add(name)
                unique_items.append(name)
            item_rows.setdefault(name, []).append((row[0].row, idx, len(parts)))

    log.info(f"Excel: {ws.max_row - header_row} data rows, {len(unique_items)} unique items")
    return unique_items, item_rows, header_row, name_col, link_col, path


def load_existing_report(report_path="processing_report.xlsx"):
    """Load previous report for resume. Returns dict: food_item -> row dict."""
    if not Path(report_path).exists():
        return {}
    try:
        wb = openpyxl.load_workbook(report_path)
        if "Report" not in wb.sheetnames:
            return {}
        ws = wb["Report"]
        headers = [str(c.value).strip() if c.value else "" for c in ws[1]]
        results = {}
        for row in ws.iter_rows(min_row=2):
            vals = [c.value for c in row]
            if not vals or not vals[0]:
                continue
            d = {}
            for h, v in zip(headers, vals):
                d[h] = v if v else ""
            food = d.get("Food Item", "")
            if food:
                results[food] = {
                    "food_item": food,
                    "search_query": d.get("Search Query Used", ""),
                    "image_found": d.get("Image Found", ""),
                    "image_processed": d.get("Image Processed", ""),
                    "uploaded": d.get("Uploaded", ""),
                    "source_url": d.get("Source URL", ""),
                    "drive_link": d.get("Drive Link", ""),
                    "quality_score": d.get("Quality Score", ""),
                    "status": d.get("Status", ""),
                    "notes": d.get("Notes", ""),
                }
        log.info(f"Loaded {len(results)} items from existing report")
        return results
    except Exception as e:
        log.warning(f"Could not read existing report: {e}")
        return {}


def write_report(report_rows, assumptions, path="processing_report.xlsx"):
    """Write / overwrite the processing report with a Report sheet and Assumptions sheet."""
    wb = openpyxl.Workbook()

    # --- Report sheet ---
    ws = wb.active
    ws.title = "Report"
    headers = ["Food Item", "Search Query Used", "Image Found", "Image Processed",
               "Uploaded", "Source URL", "Drive Link", "Quality Score", "Status", "Notes"]
    ws.append(headers)
    for r in report_rows:
        ws.append([
            r.get("food_item", ""), r.get("search_query", ""), r.get("image_found", ""),
            r.get("image_processed", ""), r.get("uploaded", ""), r.get("source_url", ""),
            r.get("drive_link", ""), r.get("quality_score", ""), r.get("status", ""),
            r.get("notes", ""),
        ])
    # Auto-width (rough)
    for col in ws.columns:
        max_len = max((len(str(c.value or "")) for c in col), default=10)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)

    # --- Assumptions sheet ---
    ws2 = wb.create_sheet("Assumptions")
    ws2.append(["#", "Assumption"])
    for i, a in enumerate(assumptions, 1):
        ws2.append([i, a])
    ws2.column_dimensions["B"].width = 100

    wb.save(path)


def write_output_excel(input_path, item_links, output_path="output.xlsx",
                       header_row=None, name_col=None, link_col=None):
    """Copy input Excel and fill in the 'Google drive link' column."""
    wb = openpyxl.load_workbook(input_path)
    ws = wb.active
    for sn in wb.sheetnames:
        if "table" in sn.lower():
            ws = wb[sn]
            break

    if header_row is None or name_col is None or link_col is None:
        # Re-detect
        for row in ws.iter_rows(min_row=1, max_row=10):
            for cell in row:
                val = str(cell.value).strip().lower() if cell.value else ""
                if val == "item_name":
                    header_row = cell.row
                    name_col = cell.column
                if "google" in val and "link" in val:
                    link_col = cell.column

    if not all([header_row, name_col, link_col]):
        log.error("Cannot locate columns for output Excel.")
        return

    for row in ws.iter_rows(min_row=header_row + 1):
        raw = row[name_col - 1].value
        if not raw:
            continue
        parts = [p.strip() for p in str(raw).split("\n") if p.strip()]
        links = [item_links.get(n, "") or "" for n in parts]
        row[link_col - 1].value = "\n".join(links)

    wb.save(output_path)
    log.info(f"Output Excel saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# NAME UNDERSTANDING
# ═══════════════════════════════════════════════════════════════════════════════

# Regex patterns for variant stripping (used in fallback)
VARIANT_RE = re.compile(
    r"\s*[\[\(]\s*(half|full|regular|special|boneless|with bone|quarter|large|small|medium"
    r"|single|double|triple|mini|family|combo)\s*[\]\)]",
    re.IGNORECASE,
)
SUFFIX_VARIANT_RE = re.compile(
    r"\s+(half|full|regular|special|boneless|with bone)\s*$",
    re.IGNORECASE,
)


def fallback_cleanup(name):
    """Regex-based name cleanup when Gemini is unavailable."""
    query = VARIANT_RE.sub("", name)
    query = SUFFIX_VARIANT_RE.sub("", query).strip()
    note = "Variant stripped" if query != name.strip() else ""
    # Add helpful context for Indian food
    if not any(w in query.lower() for w in ("recipe", "dish", "food", "indian")):
        query = query + " indian food dish"
    return {"search_query": query, "base_dish": query.replace(" indian food dish", ""), "note": note}


def batch_understand_names(names, client, model):
    """Ask Gemini to produce search queries for a batch of food item names."""
    if not client or not names:
        return {n: fallback_cleanup(n) for n in names}

    result = {}
    for i in range(0, len(names), GEMINI_BATCH_SIZE):
        batch = names[i:i + GEMINI_BATCH_SIZE]
        numbered = "\n".join(f"{j+1}. {n}" for j, n in enumerate(batch))
        prompt = (
            "You are helping prepare search queries for food photographs (mostly Indian restaurant dishes). "
            "For each food item name below, return a JSON object mapping the EXACT original name to an object "
            "with keys: search_query, base_dish, note.\n\n"
            "Rules:\n"
            "- Remove portion/variant markers like [Half], (Half), Full, Special, Boneless from search_query.\n"
            "- Add 'indian food dish' or 'restaurant style' to make the search more image-friendly.\n"
            "- base_dish = the core dish name without variants or search hints.\n"
            "- note = short note about any variant or special handling.\n"
            "- If a name looks like TWO dishes glued together (no line break), set note='Possibly two dishes'.\n\n"
            f"Items:\n{numbered}\n\n"
            "Respond with ONLY valid JSON. Keys must be the EXACT original names."
        )
        try:
            time.sleep(GEMINI_DELAY_S)
            raw = gemini_generate(client, model, [prompt], json_mode=True)
            parsed = parse_json_response(raw)
            if isinstance(parsed, dict):
                for name in batch:
                    if name in parsed and isinstance(parsed[name], dict):
                        info = parsed[name]
                        info.setdefault("search_query", name)
                        info.setdefault("base_dish", name)
                        info.setdefault("note", "")
                        result[name] = info
                    else:
                        result[name] = fallback_cleanup(name)
            else:
                for name in batch:
                    result[name] = fallback_cleanup(name)
        except Exception as e:
            log.warning(f"Gemini batch understand failed: {e}. Using fallback.")
            for name in batch:
                result[name] = fallback_cleanup(name)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def search_images(query, max_results=MAX_SEARCH_RESULTS):
    """Search DuckDuckGo for images. Returns list of result dicts."""
    try:
        results = list(DDGS().images(
            keywords=query,
            max_results=max_results,
            size="Large",
            type_image="photo",
        ))
        return [r for r in results if r.get("image")]
    except Exception as e:
        log.warning(f"Image search failed for '{query}': {e}")
        return []


def search_with_retry(query, base_dish=""):
    """Search with fallback alternate queries. Returns list of image URLs."""
    urls = []
    seen = set()

    results = search_images(query)
    for r in results:
        u = r["image"]
        if u not in seen:
            seen.add(u)
            urls.append(u)

    if len(urls) < 3:
        alternates = [f"{base_dish or query} recipe photo", f"{base_dish or query} restaurant plate"]
        for alt in alternates:
            time.sleep(SEARCH_DELAY_S)
            for r in search_images(alt, max_results=10):
                u = r["image"]
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
            if len(urls) >= 5:
                break

    return urls


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE DOWNLOAD & LOCAL QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def download_image(url, save_path):
    """Download an image URL to disk. Returns True on success."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FoodImageBot/1.0)"}
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, headers=headers, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type or "text" in content_type:
            return False

        data = b""
        for chunk in resp.iter_content(1024 * 64):
            data += chunk
            if len(data) > MAX_RAW_DOWNLOAD_BYTES:
                return False

        # Verify it's a valid image
        try:
            img = Image.open(BytesIO(data))
            img.verify()
        except Exception:
            return False

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def check_blur(image_path):
    """Return Laplacian variance (higher = sharper)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    # Resize to a standard size for consistent blur detection
    h, w = img.shape
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return cv2.Laplacian(img, cv2.CV_64F).var()


def local_quality_check(image_path):
    """Run cheap local checks. Returns (passed: bool, reason: str)."""
    try:
        img = Image.open(image_path)
        w, h = img.size
    except Exception as e:
        return False, f"Cannot open image: {e}"

    if w < MIN_SOURCE_WIDTH or h < MIN_SOURCE_HEIGHT:
        return False, f"Too small ({w}x{h}), need at least {MIN_SOURCE_WIDTH}x{MIN_SOURCE_HEIGHT}"

    variance = check_blur(image_path)
    if variance < BLUR_THRESHOLD:
        return False, f"Blurry (Laplacian variance {variance:.0f} < {BLUR_THRESHOLD})"

    return True, "OK"


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI VISION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_with_gemini(image_path, dish_name, client, model):
    """
    Ask Gemini to evaluate a food photo. Returns dict with:
      matches_dish, fully_visible, centered, too_zoomed, cropped,
      is_collage_or_has_big_text_or_watermark, quality_score (1-10),
      dish_bbox [ymin,xmin,ymax,xmax] on 0-1000 scale, reason.
    """
    default = {
        "matches_dish": True, "fully_visible": True, "centered": True,
        "too_zoomed": False, "cropped": False,
        "is_collage_or_has_big_text_or_watermark": False,
        "quality_score": 6, "dish_bbox": None, "reason": "Gemini unavailable",
    }
    if not client:
        return default

    try:
        img = Image.open(image_path).convert("RGB")
        # Resize for Gemini to save bandwidth
        w, h = img.size
        if max(w, h) > 1024:
            scale = 1024 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        prompt = (
            f"You are evaluating a photograph for use as a restaurant menu image.\n"
            f"The dish should be: {dish_name}\n\n"
            "Respond with ONLY a JSON object (no markdown):\n"
            "{\n"
            '  "matches_dish": true/false (does this show the correct dish?),\n'
            '  "fully_visible": true/false (is the full dish visible, not cut off?),\n'
            '  "centered": true/false (is the dish roughly centered?),\n'
            '  "too_zoomed": true/false (is it excessively zoomed in?),\n'
            '  "cropped": true/false (is the dish significantly cropped?),\n'
            '  "is_collage_or_has_big_text_or_watermark": true/false,\n'
            '  "quality_score": 1-10 (10=perfect menu photo),\n'
            '  "dish_bbox": [ymin, xmin, ymax, xmax] on a 0-1000 scale,\n'
            '  "reason": "short explanation"\n'
            "}"
        )
        time.sleep(GEMINI_DELAY_S)
        raw = gemini_generate(client, model, [img, prompt], json_mode=True)
        parsed = parse_json_response(raw)
        if isinstance(parsed, dict):
            for key in default:
                parsed.setdefault(key, default[key])
            return parsed
        return default
    except Exception as e:
        log.debug(f"Gemini vision evaluation failed: {e}")
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING (crop / resize / compress to 1800×1200 JPG)
# ═══════════════════════════════════════════════════════════════════════════════

def process_image(source_path, bbox, output_path):
    """
    Process source image to exactly 1800x1200 JPG under 10 MB.
    bbox: [ymin, xmin, ymax, xmax] on 0-1000 scale, or None.
    Returns (success: bool, notes: str).
    """
    try:
        img = Image.open(source_path).convert("RGB")
    except Exception as e:
        return False, f"Cannot open: {e}"

    w, h = img.size
    notes = []

    # Determine dish centre and bounds (pixels)
    if bbox and len(bbox) == 4 and all(isinstance(b, (int, float)) for b in bbox):
        ymin_f, xmin_f, ymax_f, xmax_f = [b / 1000.0 for b in bbox]
        dish_cx = (xmin_f + xmax_f) / 2 * w
        dish_cy = (ymin_f + ymax_f) / 2 * h
        dish_x1, dish_y1 = xmin_f * w, ymin_f * h
        dish_x2, dish_y2 = xmax_f * w, ymax_f * h
    else:
        # Default: assume dish is in the centre 70% of the image
        dish_cx, dish_cy = w / 2, h / 2
        dish_x1, dish_y1 = w * 0.15, h * 0.15
        dish_x2, dish_y2 = w * 0.85, h * 0.85

    # Largest 3:2 crop that fits the image
    crop_w = float(w)
    crop_h = crop_w / TARGET_RATIO
    if crop_h > h:
        crop_h = float(h)
        crop_w = crop_h * TARGET_RATIO

    # Centre on dish
    cx1 = dish_cx - crop_w / 2
    cy1 = dish_cy - crop_h / 2
    # Clamp to image edges
    cx1 = max(0.0, min(cx1, w - crop_w))
    cy1 = max(0.0, min(cy1, h - crop_h))
    cx2 = cx1 + crop_w
    cy2 = cy1 + crop_h

    # Does the dish fit inside the crop window?
    dish_fits = (dish_x1 >= cx1 - 2 and dish_y1 >= cy1 - 2 and
                 dish_x2 <= cx2 + 2 and dish_y2 <= cy2 + 2)

    if dish_fits and crop_w >= 100 and crop_h >= 100:
        cropped = img.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
        result = cropped.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        # Upscale warning
        upscale = max(TARGET_WIDTH / crop_w, TARGET_HEIGHT / crop_h)
        if upscale > UPSCALE_WARNING_FACTOR:
            notes.append(f"Low resolution source (upscaled {upscale:.1f}x)")
    else:
        # Blurred-background fallback (image too tall/narrow to crop without cutting dish)
        bg = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        bg_arr = cv2.GaussianBlur(np.array(bg), (51, 51), 30)
        bg = Image.fromarray(bg_arr)
        scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        fitted = img.resize((new_w, new_h), Image.LANCZOS)
        x_off = (TARGET_WIDTH - new_w) // 2
        y_off = (TARGET_HEIGHT - new_h) // 2
        bg.paste(fitted, (x_off, y_off))
        result = bg
        notes.append("Blurred background (unusual aspect ratio)")

    # Save as JPEG with file-size control
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    quality = JPEG_INITIAL_QUALITY
    while quality >= 20:
        buf = BytesIO()
        result.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() < MAX_FILE_SIZE_BYTES:
            with open(output_path, "wb") as f:
                f.write(buf.getvalue())
            return True, "; ".join(notes)
        quality -= 5

    # Extremely unlikely fallback
    buf = BytesIO()
    result.save(buf, format="JPEG", quality=15, optimize=True)
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())
    notes.append("Heavy compression applied")
    return True, "; ".join(notes)


# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════════════════════

def setup_drive():
    """Return (service, folder_id) or (None, None) if not configured."""
    folder_id = os.getenv("DRIVE_FOLDER_ID", "").strip()
    creds_file = os.getenv("DRIVE_CREDENTIALS_FILE", "credentials.json")
    token_file = os.getenv("DRIVE_TOKEN_FILE", "token.json")

    if not folder_id:
        log.warning("DRIVE_FOLDER_ID not set → Drive upload disabled.")
        return None, None
    if not Path(creds_file).exists():
        log.warning(f"'{creds_file}' not found → Drive upload disabled.")
        return None, None

    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request as AuthRequest
        from googleapiclient.discovery import build

        SCOPES = ["https://www.googleapis.com/auth/drive.file"]
        creds = None

        if Path(token_file).exists():
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(AuthRequest())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_file, "w") as f:
                f.write(creds.to_json())

        service = build("drive", "v3", credentials=creds)
        log.info("Google Drive connected.")
        return service, folder_id
    except Exception as e:
        log.warning(f"Drive setup failed: {e}")
        return None, None


def upload_to_drive(service, file_path, folder_id, filename):
    """Upload (or update) a file in Drive. Returns (file_id, webViewLink)."""
    from googleapiclient.http import MediaFileUpload

    for attempt in range(DRIVE_RETRY_COUNT):
        try:
            # Check for existing file with same name
            esc_name = filename.replace("'", "\\'")
            query = f"name='{esc_name}' and '{folder_id}' in parents and trashed=false"
            existing = service.files().list(q=query, fields="files(id)").execute().get("files", [])

            media = MediaFileUpload(file_path, mimetype="image/jpeg", resumable=True)

            if existing:
                fid = existing[0]["id"]
                result = service.files().update(
                    fileId=fid, media_body=media, fields="id,webViewLink"
                ).execute()
            else:
                meta = {"name": filename, "parents": [folder_id]}
                result = service.files().create(
                    body=meta, media_body=media, fields="id,webViewLink"
                ).execute()

            return result["id"], result.get("webViewLink", "")
        except Exception as e:
            if attempt < DRIVE_RETRY_COUNT - 1:
                wait = DRIVE_RETRY_BACKOFF_S * (2 ** attempt)
                log.warning(f"Drive upload retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE: per-item processing
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_candidate(image_urls, item_name, gemini_client, gemini_model):
    """
    Download and evaluate up to MAX_CANDIDATES images.
    Returns best candidate dict or None.
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    best = None
    best_score = 0

    for i, url in enumerate(image_urls[:MAX_CANDIDATES * 2]):  # try more URLs, stop at MAX_CANDIDATES evaluated
        if i > 0 and i % 3 == 0:
            time.sleep(0.5)  # polite delay
        temp_path = TEMP_DIR / f"cand_{i}.jpg"
        if not download_image(url, temp_path):
            continue

        passed, reason = local_quality_check(temp_path)
        if not passed:
            log.debug(f"  Candidate {i} local fail: {reason}")
            temp_path.unlink(missing_ok=True)
            continue

        ev = evaluate_with_gemini(temp_path, item_name, gemini_client, gemini_model)
        score = ev.get("quality_score", 0)
        if not isinstance(score, (int, float)):
            try:
                score = int(score)
            except (ValueError, TypeError):
                score = 5

        is_good = (
            ev.get("matches_dish", False)
            and ev.get("fully_visible", False)
            and not ev.get("too_zoomed", False)
            and not ev.get("cropped", False)
            and not ev.get("is_collage_or_has_big_text_or_watermark", False)
            and score >= 7
        )

        if is_good:
            # Clean up any previous best temp file
            if best and Path(best["temp_path"]).exists() and best["temp_path"] != str(temp_path):
                Path(best["temp_path"]).unlink(missing_ok=True)
            return {
                "temp_path": str(temp_path), "url": url,
                "score": score, "bbox": ev.get("dish_bbox"),
                "status": "Success",
            }

        # Track best borderline
        if score > best_score and score >= 5 and ev.get("matches_dish", False):
            if best and Path(best["temp_path"]).exists():
                Path(best["temp_path"]).unlink(missing_ok=True)
            best = {
                "temp_path": str(temp_path), "url": url,
                "score": score, "bbox": ev.get("dish_bbox"),
                "status": "Processed - Needs Review",
            }
            best_score = score
        else:
            temp_path.unlink(missing_ok=True)

    return best


def process_item(item_name, info, query_cache, gemini_client, gemini_model,
                 drive_service, drive_folder_id):
    """Run the full pipeline for one food item. Returns result dict."""
    search_query = info["search_query"]
    base_dish = info.get("base_dish", item_name)
    fname = sanitize_filename(item_name) + ".jpg"
    output_path = OUTPUT_DIR / fname

    result = {
        "food_item": item_name, "search_query": search_query,
        "image_found": "No", "image_processed": "No", "uploaded": "No",
        "source_url": "", "drive_link": "", "quality_score": "",
        "status": "Failed - No Suitable Image", "notes": info.get("note", ""),
    }

    # ── Cache hit (same search_query already succeeded) ──
    cached = query_cache.get(search_query)
    if cached and Path(cached["processed_path"]).exists():
        shutil.copy2(cached["processed_path"], output_path)
        result.update({
            "image_found": "Yes", "image_processed": "Yes",
            "source_url": cached["source_url"], "quality_score": cached["quality_score"],
            "status": cached["status"],
        })
        result["notes"] = (result["notes"] + " Reused cached image.").strip()
        _do_upload(result, output_path, drive_service, drive_folder_id)
        log.info(f"  ✓ {item_name}  (cached)")
        return result

    # ── Search ──
    time.sleep(SEARCH_DELAY_S)
    image_urls = search_with_retry(search_query, base_dish)
    if not image_urls:
        log.warning(f"  ✗ {item_name}  No images found")
        return result

    # ── Evaluate candidates ──
    best = find_best_candidate(image_urls, item_name, gemini_client, gemini_model)
    if not best:
        log.warning(f"  ✗ {item_name}  No suitable candidate")
        return result

    result["image_found"] = "Yes"
    result["source_url"] = best["url"]
    result["quality_score"] = best["score"]

    # ── Process image ──
    success, proc_notes = process_image(best["temp_path"], best.get("bbox"), output_path)
    Path(best["temp_path"]).unlink(missing_ok=True)

    if not success:
        result["status"] = "Failed - Processing Error"
        result["notes"] = (result["notes"] + " " + proc_notes).strip()
        log.warning(f"  ✗ {item_name}  Processing error: {proc_notes}")
        return result

    result["image_processed"] = "Yes"
    result["status"] = best["status"]
    if proc_notes:
        result["notes"] = (result["notes"] + " " + proc_notes).strip()

    # ── Cache for items sharing the same search query ──
    query_cache[search_query] = {
        "processed_path": str(output_path),
        "source_url": best["url"], "quality_score": best["score"],
        "status": best["status"],
    }

    # ── Upload ──
    _do_upload(result, output_path, drive_service, drive_folder_id)

    status_icon = "✓" if result["status"] == "Success" else "⚠"
    log.info(f"  {status_icon} {item_name}  score={best['score']}  {result['uploaded']}")
    return result


def _do_upload(result, output_path, drive_service, drive_folder_id):
    """Upload to Drive and update result dict in-place."""
    if drive_service and drive_folder_id:
        try:
            _, link = upload_to_drive(drive_service, str(output_path), drive_folder_id, output_path.name)
            result["uploaded"] = "Yes"
            result["drive_link"] = link
        except Exception as e:
            log.error(f"Drive upload failed: {e}")
            result["uploaded"] = "No - Upload Failed"
            result["notes"] = (result.get("notes", "") + f" Drive error: {e}").strip()
    else:
        result["uploaded"] = "No - Drive not configured"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    load_dotenv()
    setup_logging()

    log.info("=" * 60)
    log.info("  Food Image Automation Agent")
    log.info("=" * 60)

    # 1. Read Excel
    unique_items, item_rows, header_row, name_col, link_col, input_path = read_excel(args.input)

    if args.limit:
        unique_items = unique_items[: args.limit]
        log.info(f"Limiting to first {args.limit} unique items")

    # 2. Resume support
    existing = {}
    if not args.force:
        existing = load_existing_report()

    skip_items = {n for n, r in existing.items() if str(r.get("uploaded", "")).strip().lower() == "yes"}

    # 3. Gemini setup
    gemini_client, gemini_model = setup_gemini()

    # 4. Understand names
    to_understand = [n for n in unique_items if n not in skip_items]
    name_info = batch_understand_names(to_understand, gemini_client, gemini_model)
    for n in unique_items:
        if n not in name_info:
            name_info[n] = existing.get(n, {}).get("search_query") and {
                "search_query": existing[n]["search_query"],
                "base_dish": existing[n]["search_query"],
                "note": "",
            } or fallback_cleanup(n)

    # 5. Drive setup
    drive_service, drive_folder_id = setup_drive()

    # 6. Process items
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    report_data = dict(existing)  # start from existing for resume
    item_links = {n: r.get("drive_link", "") for n, r in existing.items()}
    query_cache = {}

    stats = {"success": 0, "review": 0, "failed": 0, "skipped": 0, "uploaded": 0}

    pbar = tqdm(unique_items, desc="Processing", unit="item")
    for item_name in pbar:
        short = item_name[:30]
        pbar.set_postfix_str(short, refresh=False)

        if item_name in skip_items:
            stats["skipped"] += 1
            continue

        try:
            result = process_item(
                item_name, name_info[item_name], query_cache,
                gemini_client, gemini_model, drive_service, drive_folder_id,
            )
        except Exception as e:
            log.error(f"ITEM EXCEPTION '{item_name}': {e}", exc_info=True)
            result = {
                "food_item": item_name, "search_query": name_info.get(item_name, {}).get("search_query", ""),
                "image_found": "No", "image_processed": "No", "uploaded": "No",
                "source_url": "", "drive_link": "", "quality_score": "",
                "status": "Failed - Error", "notes": str(e),
            }

        report_data[item_name] = result
        item_links[item_name] = result.get("drive_link", "")

        # Stats
        st = result.get("status", "")
        if st == "Success":
            stats["success"] += 1
        elif "Review" in st:
            stats["review"] += 1
        else:
            stats["failed"] += 1
        if result.get("uploaded") == "Yes":
            stats["uploaded"] += 1

        # Save report after EVERY item (crash safe)
        ordered_rows = [report_data[n] for n in unique_items if n in report_data]
        write_report(ordered_rows, ASSUMPTIONS)

    # Clean up temp dir
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # 7. Write output.xlsx
    write_output_excel(input_path, item_links, "output.xlsx", header_row, name_col, link_col)

    # 8. Summary
    total = len(unique_items)
    print(f"\n{'=' * 60}")
    print(f"  Processing Complete!")
    print(f"  Total unique items : {total}")
    print(f"  Skipped (resumed)  : {stats['skipped']}")
    print(f"  Success            : {stats['success']}")
    print(f"  Needs Review       : {stats['review']}")
    print(f"  Failed             : {stats['failed']}")
    print(f"  Uploaded to Drive  : {stats['uploaded']}")
    print(f"{'=' * 60}")
    print(f"  Reports: processing_report.xlsx, output.xlsx")
    print(f"  Images:  {OUTPUT_DIR}/")
    print(f"  Log:     agent.log")


if __name__ == "__main__":
    main()
