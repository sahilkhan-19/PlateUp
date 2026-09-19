# AI Agent for Automated Food Image Collection & Processing

An AI-powered automation agent that reads food item names from an Excel file,
finds suitable images online, validates quality with Gemini Vision, processes
them to exact specifications, and uploads to Google Drive.

```
Excel → Understand Name → Image Search → Evaluate → Process (1800×1200) → Rename → Google Drive
```

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a Gemini API key**
   - Go to [Google AI Studio](https://aistudio.google.com/apikey)
   - Create a new API key
   - Copy it — you'll need it for `.env`

4. **Enable the Google Drive API**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create or select a project
   - Enable the **Google Drive API**
   - Go to **Credentials → Create Credentials → OAuth client ID**
   - Application type: **Desktop app**
   - Download the JSON and save it as `credentials.json` in this folder

5. **Create a target folder in Google Drive**
   - Create a folder in Google Drive for the output images
   - Open it and copy the folder ID from the URL:
     `https://drive.google.com/drive/folders/<THIS_IS_THE_FOLDER_ID>`

6. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in:
   - `GEMINI_API_KEY` — from step 3
   - `DRIVE_FOLDER_ID` — from step 5

## Running

### Quick test (5 items)
```bash
python agent.py --input input.xlsx --limit 5
```

### Full run (all ~260 unique items)
```bash
python agent.py --input input.xlsx
```

### Force re-process everything
```bash
python agent.py --input input.xlsx --force
```

### Test with the sample file
```bash
python agent.py --input sample_input.xlsx
```

### What you will see
- A progress bar with the current item name
- One log line per item showing ✓ (success), ⚠ (needs review), or ✗ (failed)
- On first Drive use, a browser window opens for Google OAuth — sign in and authorize
- After completion: a summary table with counts
- `processing_report.xlsx` is saved after **every** item, so progress is never lost

## Outputs

| File | Description |
|------|-------------|
| `output/images/*.jpg` | Processed images (1800×1200, JPG, <10 MB) |
| `output.xlsx` | Copy of input with Google Drive links filled in |
| `processing_report.xlsx` | Per-item status report + assumptions sheet |
| `agent.log` | Detailed debug log |

## How Image Validation Works

Each candidate image goes through **two stages** of validation:

1. **Local checks (fast, no API):** reject images smaller than 900×600 px or blurry
   (Laplacian variance < 100).
2. **Gemini Vision check:** the LLM confirms the image matches the dish, is fully visible,
   properly centred, not a collage/watermarked, and assigns a quality score (1-10).
   Images scoring ≥ 7 with all checks passing are accepted. If none pass, the best image
   with score ≥ 5 is used and flagged "Needs Review".

## Known Limitations

- DuckDuckGo image search may rate-limit after many queries; the agent adds delays and retries
  but very large runs may need to be done in batches.
- Gemini vision evaluation depends on API availability; if the API is down, a basic fallback
  (local checks only) is used and items are flagged for review.
- Some compound names without line breaks (e.g., "Butter Naan Chicken Crispy") may not be
  correctly identified as two separate dishes.
- Images are from public web search and may have implicit copyright — intended for internal
  restaurant-menu use only.

## Assumptions

1. Duplicate item names are processed once and reused for every row that has that name.
2. A cell with two dishes separated by a line break is treated as two separate items.
3. Variant names ([Half], (Half), Full, Special, etc.) are searched using the base dish, but the
   saved file keeps the exact name from Excel.
4. Names that look like two dishes glued together with no line break are searched as written and
   may be flagged "Needs Review".
5. Images come from public web search (DuckDuckGo) and are for internal restaurant-menu use; the
   source URL of every image is logged in the report. Minimum accepted source size is 900×600 px.
6. If no image passes all checks, the best borderline image is used and flagged for manual review.
7. Output format is JPG at 1800×1200 (3:2), quality ~90, always under 10 MB.
8. The "Google drive link" column of the input sheet is filled in output.xlsx (the original file
   is not modified).

## Creating submission.zip

```bash
python -c "import shutil; shutil.make_archive('submission', 'zip', '.', '.')" 
```

Or use the targeted command:
```bash
python -m zipfile -c submission.zip agent.py requirements.txt .env.example README.md input.xlsx sample_input.xlsx output.xlsx processing_report.xlsx submission_notes.txt
```

This excludes `.env`, `credentials.json`, `token.json`, the venv, and `output/images/`
(images live in Google Drive).
