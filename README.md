# Metadata-AI

Metadata-AI is a local AI-powered tool that automatically tags and dates scanned physical photographs by writing metadata directly into image files.

It uses a vision language model (VLM) running locally via [LM Studio](https://lmstudio.ai) to analyze each photo, detect whether the next scanned image is the back of a photograph, extract handwritten dates and comments via OCR (translating to English if needed), and estimate dates from visual cues like fashion and technology when no written date is available.

---

## Features

- Detects back-of-photo scans and extracts handwritten dates via OCR
- Extracts and saves handwritten comments from the back of photos, translating to English if needed
- Falls back to AI visual date estimation based on fashion and technology in the image
- Writes `DateTimeOriginal` and keyword tags into EXIF metadata
- Supports `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.png`, `.heic`, `.webp`, and `.dng` files
- DNG files are written as `.xmp` sidecar files, compatible with Lightroom and Apple Photos
- All images are converted to JPEG and downscaled to a maximum of 2048px on the long edge before being sent to LM Studio — keeping API calls fast while staying within LM Studio's JPEG/PNG/WebP support
- Handles very large scanned photos (600–1200 DPI scans can exceed 100 MP) without crashing
- Skips photos dated at or after a configurable cutoff year (default: 2010)
- Runs entirely locally — no cloud API required
- Prompts for photos directory at runtime, or accepts it as a command line argument

---

## Requirements

- Python 3.8+
- [LM Studio](https://lmstudio.ai) running locally with a vision-capable model loaded

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
```

Or pass the directory as a command line argument:

```bash
python metadata-ai.py /path/to/photos
```

If you scanned the backs of photos, place them immediately after the front in filename order (e.g. `img001.jpg` front, `img002.jpg` back). The script will detect and pair them automatically.

---

## How It Works

1. **Back detection** — For each photo, the script checks if the next image is the reverse side of a physical print using a two-step VLM confirmation. If confirmed, it attempts to OCR a date and any handwritten comments from the back. Comments are translated to English if needed. If no date is found on the back, the script falls through to step 3.
2. **Existing IPTC check** — If no back is found, it checks for an existing date in the photo's IPTC metadata.
3. **AI visual estimation** — If still no date, the VLM analyzes fashion, technology, and other visual cues to estimate the year and month.
4. **Metadata writing** — Valid dates before the cutoff year are written into the file along with AI-generated keyword tags and any comments extracted from the back. DNG files receive a `.xmp` sidecar instead of direct EXIF modification.

---

## Supported Formats

| Format | Metadata method |
|---|---|
| `.jpg`, `.jpeg` | piexif (EXIF) + iptcinfo3 (IPTC) |
| `.tiff`, `.tif` | Pillow (TIFF tags) |
| `.png` | Pillow (TIFF tags) |
| `.heic` | Pillow via pillow-heif |
| `.webp` | Pillow |
| `.dng` | XMP sidecar file |

---

## License

MIT
