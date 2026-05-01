# Metadata AI

Metadata AI is a local AI-powered tool that automatically tags and dates scanned physical photographs by writing metadata directly into JPEG and TIFF EXIF data.

It uses a vision language model (VLM) running locally via [LM Studio](https://lmstudio.ai) to analyze each photo, detect whether the next scanned image is the back of a photograph, extract handwritten dates via OCR, and estimate dates from visual cues like fashion and technology when no written date is available.

---

## Features

- Detects back-of-photo scans and extracts handwritten dates via OCR
- Falls back to AI visual date estimation based on fashion and technology in the image
- Writes `DateTimeOriginal`, comments from the back of the photo, and keyword tags into EXIF metadata
- Supports `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.png`, and `.heic` files
- Skips photos dated 2010 or later. This can be customized in the script.
- Runs entirely locally — no cloud API required

---

## Requirements

- Python 3.8+
- [LM Studio](https://lmstudio.ai) running locally with a vision-capable model loaded

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/bcholodenko/Metadata-AI.git
   cd metadata-ai
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
DIRECTORY = "./photos"   # Path to your folder of scanned images
MODEL_ID = "qwen/qwen3.6-27b"  # Must match the model identifier in LM Studio
```

---

## Usage

Place your scanned photos in the `./photos` folder (or update `DIRECTORY`), then run:

```bash
python metadata-ai.py
```

If you scanned the backs of photos, place them immediately after the front in filename order (e.g. `img001.jpg` front, `img002.jpg` back). The script will detect and pair them automatically.

---

## How It Works

1. **Back detection** — For each photo, the script checks if the next image is the reverse side of a physical print. If confirmed, it attempts to OCR a date from the back. If no date is found, the photo is skipped.
2. **Existing EXIF check** — If no back is found, it checks for an existing date in the photo's EXIF UserComment field.
3. **AI visual estimation** — If still no date, the VLM analyzes fashion, technology, and other visual cues to estimate the year and month.
4. **Metadata writing** — Valid dates are written into the file's EXIF data along with AI-generated keyword tags.

---

## License

MIT
