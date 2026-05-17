<img width="600" alt="logo" src="https://github.com/user-attachments/assets/8bcf39fd-93c3-4017-96e0-7d692237b197" />

Automatically tag and date scanned photos and home videos using a **local vision-language model (VLM)** running in [LM Studio](https://lmstudio.ai). No data leaves your machine.

Metadata-AI analyses each photo through a five-step pipeline — checking the back of the photo, existing IPTC keywords, and a VLM date estimate — then writes the result directly into the file's EXIF/XMP metadata or a sidecar. Low-confidence guesses are queued for a fast interactive review rather than written blindly.

---

## Features

- **Automatic dating** — estimates photo dates from fashion, hairstyles, technology, and setting visible in the image; supports exact dates, decades, and fuzzy expressions like *"circa 1965"* or *"early 1980s"*
- **Keyword tagging** — generates 5–8 descriptive keywords per photo and writes them as IPTC/XMP subject tags
- **Scene description** — one-sentence scene summary stored in the image description field
- **Indoor / outdoor detection** and **flash detection**
- **Geotagging** — optional location identification via Nominatim (OpenStreetMap); confidence-gated to avoid landscape guesses
- **Video analysis** — frame-by-frame VLM analysis with consensus dating, structured metadata, and an ffmpeg write-back
- **Folder consensus** — low-confidence dates in a folder are corrected to the folder's majority year when a clear consensus exists
- **Interactive review queue** — uncertain photos are written to `review.json` / `review.html` for a fast accept / edit / skip pass
- **Resumable sessions** — a checkpoint file lets interrupted runs pick up exactly where they left off
- **Dry-run mode** — preview every action without touching any file
- **Batch processing** — process multiple folders in one run; recursive subfolder support
- **XMP sidecar mode** — write `.xmp` files only, leaving originals untouched (always used for DNG and RAW files)
- **RAW format support** — CR2, CR3, NEF, ARW, RAF, ORF, RW2, DNG (requires `rawpy`)
- **Back-of-photo detection** — reads handwritten dates from the back of scanned prints via the VLM or optional Tesseract OCR

---

## Requirements

### Python

Python 3.10 or later.

### Python packages

```bash
pip install -r requirements.txt
```

Core dependencies installed automatically:

| Package | Purpose |
|---|---|
| `openai` | LM Studio API client |
| `Pillow` | Image loading and resizing |
| `pillow-heif` | HEIC / HEIF support |
| `iptcinfo3` | Existing IPTC keyword reading |
| `natsort` | Natural filename ordering |
| `rich` | Terminal UI (progress bars, panels) |

Optional packages (uncomment in `requirements.txt` to enable):

| Package | Feature unlocked |
|---|---|
| `geopy` | Geotagging via Nominatim (`--geotag`) |
| `rawpy` + `numpy` | RAW camera file processing |
| `questionary` | Arrow-key interactive menus |
| `pytesseract` | OCR on photo backs |

### External binaries

**ExifTool** — merges XMP metadata into JPEG, TIFF, PNG, HEIC, and WebP files.

```bash
# macOS
brew install exiftool

# Ubuntu / Debian
sudo apt install libimage-exiftool-perl

# Windows — download installer from https://exiftool.org
```

**ffmpeg** — required for video frame extraction and metadata write-back.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

### LM Studio

1. Download and install [LM Studio](https://lmstudio.ai).
2. Load a vision-capable model. The default is `qwen/qwen3.6-27b`; any OpenAI-compatible VLM works.
3. Start the **Local Server** (default: `http://localhost:1234`).

---

## Installation

```bash
git clone https://github.com/bcholodenko/Metadata-AI.git
cd Metadata-AI
pip install -r requirements.txt
```

---

## Configuration

Copy or edit `configuration.json` (all fields are optional — built-in defaults are used for anything omitted):

```json
{
    "lm_studio": {
        "url":   "http://localhost:1234/v1",
        "model": "qwen/qwen3.6-27b"
    },
    "defaults": {
        "directory":       "./photos",
        "cutoff_year":     2010,
        "date_confidence": 7,
        "geo_confidence":  7
    },
    "limits": {
        "vlm_max_dimension":  2048,
        "max_image_pixels":   500000000,
        "min_photo_year":     1826,
        "min_video_year":     1888,
        "max_year":           2100
    }
}
```

| Key | Description |
|---|---|
| `lm_studio.url` | Base URL of the LM Studio local server |
| `lm_studio.model` | Model ID as shown in LM Studio |
| `defaults.cutoff_year` | Photos from this year or later are skipped (they're probably already dated) |
| `defaults.date_confidence` | Minimum VLM confidence (1–10) to write a date without queuing for review |
| `defaults.geo_confidence` | Minimum confidence to accept a location guess and geocode it |
| `limits.vlm_max_dimension` | Longest image side (px) sent to the VLM — larger images are downscaled |

---

## Usage

### Fully interactive (recommended for first use)

```bash
python metadata-ai.py
```

You will be prompted for the directory, cutoff year, confidence threshold, and feature flags.

### Supply the directory, prompt for the rest

```bash
python metadata-ai.py /path/to/scanned/photos
```

### Non-interactive / scripted

```bash
# Recursive, geotagging on, folder-consensus year correction
python metadata-ai.py /path/to/photos -r --geotag --consensus

# Strict confidence, XMP sidecars only, dry run first
python metadata-ai.py /path/to/photos --cutoff 1995 --date-confidence 8 --xmp-only --dry-run

# Single file
python metadata-ai.py /path/to/photo.jpg

# Video file
python metadata-ai.py /path/to/video.mp4 --video-interval 30
```

### CLI reference

```
positional arguments:
  directory             Path to a directory or single file (prompted if omitted)

options:
  --cutoff YEAR         Skip photos from this year or later (default: 2010)
  --date-confidence 1-10
                        Minimum date confidence to write without review (default: 7)
  --xmp-only            Write XMP sidecar files only; skip ExifTool merge
  --geotag              Enable geotagging via Nominatim
  --geo-confidence 1-10
                        Minimum location confidence before geocoding (default: 7)
  -r, --recursive       Recursively process all subfolders
  --consensus           Use folder consensus year to correct low-confidence dates
  --dry-run             Preview actions without modifying any files
  --skip-dated          Skip photos that already have a DateTimeOriginal tag
  --review              Run interactive review for pending items in review.json
  --model MODEL         LM Studio model ID to use
  --video-interval SEC  Seconds between sampled frames for video analysis (default: 30)
  --output PATH         Output .txt path for video analysis report
```

---

## How It Works

Each photo passes through a five-step pipeline:

1. **Back-of-photo detection** — if the next file in filename order appears to be a scanned back (handwriting, no scene), its text is read via the VLM (or Tesseract OCR if installed) and parsed for a date.
2. **IPTC keyword check** — existing keywords are scanned for parseable dates, which are used directly if found.
3. **VLM date estimation** — the image is sent to the local VLM with a prompt asking it to estimate the date from visual cues (fashion, technology, setting). A second call extracts time of day, scene description, setting, flash, location, and keywords.
4. **Location resolution** — if geotagging is enabled and the VLM identified a specific location with sufficient confidence, Nominatim converts it to GPS coordinates.
5. **Write decision** — dates at or above the confidence threshold are written immediately; dates below threshold (or with no date at all) are added to the review queue.

After all photos in a folder are processed, an optional **consensus pass** can rewrite low-confidence dates to the folder's majority year when a clear majority exists (more than 50% of high-confidence results agree).

### Review queue

Photos that cannot be dated confidently are written to `review.json` and a dark-mode `review.html` (with thumbnails) inside the processed folder. Run the interactive review at any time:

```bash
python metadata-ai.py /path/to/photos --review
```

Each photo offers four choices: **accept** the AI's guess, **enter** a different date, **skip** permanently, or **quit** (progress is saved and the session can be resumed).

### Supported formats

**Photos:** JPEG, TIFF, PNG, HEIC, WebP, DNG, CR2, CR3, NEF, ARW, RAF, ORF, RW2, RAW

**Videos:** MP4, MOV, AVI, M4V, MKV, MTS, M2TS, WMV, FLV, WebM

---

## Output files

| File | Description |
|---|---|
| `review.json` | Machine-readable review queue for the folder |
| `review.html` | Browser-viewable review page with thumbnails and status indicators |
| `metadata-ai-report.html` | Dark-mode run summary: decade chart, keyword cloud, top locations, per-folder stats |
| `<video>_summary.txt` | Plain-text frame-by-frame analysis and summary for video files |
| `.metadata-ai-progress` | Checkpoint file used to resume interrupted runs (auto-deleted on completion) |
| `*.xmp` | XMP sidecar files (created for RAW/DNG files, or when `--xmp-only` is set) |

---

## Tips

- **Start with `--dry-run`** to see what would be written before committing to any changes.
- **Set `--cutoff`** to the year your digital camera era begins — photos from that year onward are skipped because they already carry accurate EXIF dates.
- **Lower `--date-confidence`** (e.g. `5`) to write more dates automatically; raise it (e.g. `9`) to send more photos to the review queue.
- **Folder names help** — folders named `"Summer 1975"` or `"June 1962 Vacation"` are detected and passed to the VLM as high-confidence date/location hints.
- **Use `--consensus`** for rolls of film or batches scanned from the same occasion — the VLM's majority opinion corrects outliers.
- **RAW files always get XMP sidecars** regardless of the `--xmp-only` flag, since writing directly into RAW containers is unsafe.
- **Pause mid-run** by typing `p` + Enter in the terminal; the current photo finishes before pausing. Press Enter to resume, or Ctrl-C to quit (checkpoint is saved).

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Contributing

Issues and pull requests are welcome at [github.com/bcholodenko/Metadata-AI](https://github.com/bcholodenko/Metadata-AI)
