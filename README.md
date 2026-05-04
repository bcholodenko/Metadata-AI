<img width="600" alt="logo" src="https://github.com/user-attachments/assets/8bcf39fd-93c3-4017-96e0-7d692237b197" />

Metadata-AI is a local AI-powered tool that automatically tags and dates scanned physical photographs by writing metadata directly into image files.

It uses a vision language model (VLM) running locally via [LM Studio](https://lmstudio.ai) to analyze each photo, detect whether the next scanned image is the back of a photograph, extract handwritten dates and comments via OCR (translating to English if needed), and estimate dates from visual cues like fashion and technology when no written date is available.

---

## Features

- Detects back-of-photo scans and extracts handwritten dates via OCR
- Extracts and saves handwritten comments from the back of photos, translating to English if needed
- Passes the folder name to the VLM as high-confidence date and location information (e.g. `8-79 Hawaii` → August 1979, Hawaii)
- Single VLM call per photo extracts date, time of day, scene description, indoor/outdoor setting, flash, keywords, and location — minimizing processing time
- Writes `DateTimeOriginal` into EXIF and keywords/captions into IPTC metadata
- Optionally geotags photos by identifying locations from visual clues and resolving GPS coordinates via Nominatim
- Low-confidence date estimates are skipped and logged to a `review.html` report for manual review
- Optional folder consensus mode uses the most common year across high-confidence results to correct low-confidence estimates in the same folder
- Handles VLM date ranges (e.g. `1975–1978`) by using the midpoint year
- Supports `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.png`, `.heic`, `.webp`, `.dng`, and raw formats (`.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`, `.raw`)
- DNG files are written as `.xmp` sidecar files, compatible with Lightroom and Apple Photos
- All images are converted to JPEG and downscaled to a maximum of 2048px on the long edge before being sent to LM Studio — keeping API calls fast while staying within LM Studio's JPEG/PNG/WebP support
- Handles very large scanned photos (600–1200 DPI scans can exceed 100 MP) without crashing
- Skips photos dated at or after a configurable cutoff year (default: 2010)
- Can recursively process all subfolders within a directory
- Runs entirely locally — no cloud API required
- Prompts for photos directory at runtime, or accepts it as a command line argument

---

## Requirements

- Python 3.8+
- [LM Studio](https://lmstudio.ai) running locally with a vision-capable model loaded
- [ExifTool](https://exiftool.org) for writing metadata into all image formats (`brew install exiftool`)
- [rawpy](https://letmaik.github.io/rawpy/) for opening raw camera formats (`pip install rawpy`)

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

   > **Note:** HEIC support requires `pillow-heif`, which is included in `requirements.txt`. On some systems you may also need `libheif` installed via Homebrew: `brew install libheif`

   > **Optional:** Install `pytesseract` and [Tesseract](https://github.com/tesseract-ocr/tesseract) to enable OCR on the backs of photos for handwritten date extraction. Without it, the VLM handles back-of-photo text alone. To enable: `pip install pytesseract` and `brew install tesseract`

3. Open LM Studio, load a vision-capable model, and start the local server (default: `http://localhost:1234`).

---

## Configuration

At the top of `metadata-ai.py`, set the following:

```python
DIRECTORY = "./photos"         # Default fallback if no directory is provided at runtime
MODEL_ID = "qwen/qwen3.6-27b" # Must match the model identifier in LM Studio
VLM_MAX_DIMENSION = 2048       # Long-edge pixel cap before sending to VLM
```

The cutoff year and other options are set interactively at runtime.

---

## Usage

Run the script — you will be prompted for the photos directory and processing options:

```bash
python metadata-ai.py
```

Example prompts:
```
Enter photos directory [./photos]: /Users/johnappleseed/Pictures/Scans
Skip photos dated from which year or later? [2010]: 1995
Confidence threshold (1-10) [7]: 7
Write metadata to XMP sidecar files only? [y/N]: n
Enable geotagging? [y/N]: n
Recursively process subfolders? [y/N]: y
Average dates in each folder using consensus year? [y/N]: y
```

Or pass the directory as a command line argument:

```bash
python metadata-ai.py /path/to/photos
```

If you scanned the backs of photos, place them immediately after the front in filename order (e.g. `img001.jpg` front, `img002.jpg` back). The script will detect and pair them automatically.

---

## How It Works

1. **Back detection** — For each photo, the script checks if the next image is the reverse side of a physical print using a two-step VLM confirmation. If confirmed, it attempts to OCR a date and any handwritten comments from the back. Comments are translated to English if needed. If no date is found on the back, the script falls through to step 2.
2. **IPTC keyword check** — Checks existing IPTC keywords on the photo for a parseable date (e.g. "Sep 1960" or "circa 1975").
3. **AI analysis** — A single VLM call analyzes the image and returns all of the following in one pass: date estimate and confidence (if date is still unknown), time of day from lighting and shadows, a one-sentence scene description, indoor/outdoor setting, whether flash fired, location clues (if geotagging is enabled), and 5 descriptive keywords. The folder name is always passed as high-confidence date and location information for the VLM to weight accordingly. The estimated time is written into the `DateTimeOriginal` EXIF timestamp.
4. **Geotagging** — If enabled, uses the location returned by the VLM in step 3 and resolves it to GPS coordinates via Nominatim.
5. **Keywords** — Confirmed from the VLM's step 3 response — no additional call needed.
6. **Metadata writing** — Valid dates before the cutoff year are written into the file along with the generated keywords and any comments extracted from the back. If folder consensus mode is enabled, writes are deferred until all photos in the folder are analyzed — the most common year among high-confidence results is then applied to low-confidence photos (keeping their individual month/day/time) before writing. Low-confidence estimates with no consensus available are added to a `review.html` report. DNG and raw files receive a `.xmp` sidecar instead of direct EXIF modification.

---

## Supported Formats

| Format | Metadata method |
|---|---|
| `.jpg`, `.jpeg` | XMP sidecar merged into EXIF via ExifTool (sidecar deleted on success) |
| `.tiff`, `.tif` | XMP sidecar merged into EXIF via ExifTool (sidecar deleted on success) |
| `.png` | XMP sidecar merged into EXIF via ExifTool (sidecar deleted on success) |
| `.heic`, `.webp` | XMP sidecar merged into EXIF via ExifTool (sidecar deleted on success) |
| `.dng` | XMP sidecar file |
| `.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`, `.raw` | Decoded via rawpy for VLM preview; XMP sidecar file |

---

## License

MIT
