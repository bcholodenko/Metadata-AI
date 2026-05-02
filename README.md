# Metadata-AI

<img width="600" alt="logo" src="https://github.com/user-attachments/assets/0136052d-84d7-48e3-8967-a2b09c5c38e2" />

Metadata-AI is a local AI-powered tool that automatically tags and dates scanned physical photographs by writing metadata directly into image files.

It uses a vision language model (VLM) running locally via [LM Studio](https://lmstudio.ai) to analyze each photo, detect whether the next scanned image is the back of a photograph, extract handwritten dates and comments via OCR (translating to English if needed), and estimate dates from visual cues like fashion and technology when no written date is available.

---

## Features

- Detects back-of-photo scans and extracts handwritten dates via OCR
- Extracts and saves handwritten comments from the back of photos, translating to English if needed
- Handles fuzzy/vague dates like "circa 1950s" or "early 1970s", mapping them to the start of that range while preserving the raw text in `UserComment`
- Falls back to AI visual date estimation based on fashion, hairstyles, and technology in the image
- Confidence scoring — low-confidence guesses are sidelined to an HTML review report instead of being written automatically
- Optional Tesseract OCR pre-processing for improved accuracy on faded or cursive handwriting
- Optional geotagging via Nominatim — identifies locations from visual landmarks and back-of-photo text, writes GPS coordinates to EXIF
- Writes `DateTimeOriginal` and keyword tags into EXIF metadata
- Supports `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.png`, `.heic`, `.webp`, and `.dng` files
- Optional XMP sidecar mode — writes all metadata to `.xmp` files instead of modifying originals
- DNG files always use XMP sidecars, compatible with Lightroom and Apple Photos
- All images are converted to JPEG in memory before being sent to LM Studio (LM Studio supports JPEG, PNG, and WebP only)
- Skips photos dated at or after a configurable cutoff year (default: 2010)
- Runs entirely locally — no cloud API required
- Prompts for photos directory at runtime, or accepts it as a command line argument

---

## Requirements

- Python 3.8+
- [LM Studio](https://lmstudio.ai) running locally with a vision-capable model loaded
- Tesseract (optional, for improved OCR): `brew install tesseract`
- libheif (required for HEIC support): `brew install libheif`

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/bcholodenko/Metadata-AI.git
   cd Metadata-AI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open LM Studio, load a vision-capable model, and start the local server (default: `http://localhost:1234`).

---

## Configuration

At the top of `metadata-ai.py`, set the following:

```python
DIRECTORY = "./photos"         # Default fallback if no directory is provided at runtime
MODEL_ID = "qwen/qwen3.6-27b" # Must match the model identifier in LM Studio
```

All other options are set interactively at runtime.

---

## Usage

Run the script — you will be prompted for all options:

```bash
python metadata-ai.py
```

Or pass the directory as a command line argument:

```bash
python metadata-ai.py /path/to/photos
```

### Runtime prompts

```
Enter photos directory [./photos]: /Volumes/Pictures/Family
Skip photos dated from which year or later? [2010]: 
Confidence threshold for auto-write (1-10) [7]: 
Write metadata to XMP sidecar files only? [y/N]: 
Enable geotagging? [y/N]: 
```

Just press Enter to accept the default for any option.

If you scanned the backs of photos, place them immediately after the front in filename order (e.g. `img001.jpg` front, `img002.jpg` back). The script will detect and pair them automatically.

---

## How It Works

1. **Back detection** — For each photo, the script checks if the next image is the reverse side of a physical print using a two-step VLM confirmation. If Tesseract is installed, it runs a first-pass OCR to feed context into the VLM prompt. The VLM then extracts the date and any handwritten comments, translating to English if needed.
2. **Fuzzy date handling** — Vague dates like "circa 1950s" or "late 1960s" are mapped to the start of that range (e.g. `1950:01:01`) while the raw text is preserved in `UserComment`.
3. **Existing EXIF check** — If no back is found, the script checks for an existing date in the photo's EXIF `UserComment` field.
4. **AI visual estimation** — If still no date, the VLM analyzes fashion, technology, and other visual cues to estimate the date, and provides a confidence score (1–10).
5. **Confidence gating** — Results at or above the threshold (default 7/10) are written automatically. Low-confidence results are added to a `review.html` report in the photos folder for manual review.
6. **Geotagging** (optional) — The VLM looks for location clues in the image and back text. If a location is identified, Nominatim resolves it to GPS coordinates written into EXIF. If coordinates can't be resolved, the location name is stored as a text tag instead.
7. **Metadata writing** — Valid dates are written into the file along with keyword tags, comments, raw date text, and GPS coordinates. DNG files and XMP-only mode write to `.xmp` sidecars instead.

---

## Review Report

When any photo scores below the confidence threshold, a `review.html` file is generated in the photos directory at the end of the run. It lists the filename, VLM date guess, confidence score, and any extracted comments for each photo needing review.

---

## Supported Formats

| Format | Metadata method |
|---|---|
| `.jpg`, `.jpeg` | piexif (EXIF) |
| `.tiff`, `.tif` | Pillow (TIFF tags) |
| `.png` | Pillow (TIFF tags) |
| `.heic` | Pillow via pillow-heif |
| `.webp` | Pillow |
| `.dng` | XMP sidecar file |

---

## License

MIT
