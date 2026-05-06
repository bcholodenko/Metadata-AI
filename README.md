<img width="600" alt="logo" src="https://github.com/user-attachments/assets/8bcf39fd-93c3-4017-96e0-7d692237b197" />
# Metadata-AI

Metadata-AI is a local, AI-powered tool that automatically tags and dates scanned physical photographs and home video by writing metadata directly into image EXIF/XMP and video container fields.

It runs a vision language model locally via [LM Studio](https://lmstudio.ai), so nothing leaves your machine. For each photo it can detect whether the next scan is the back of a print, OCR any handwritten dates and captions, fall back to estimating a date from visual cues (fashion, hairstyles, technology, setting), and produce a structured metadata record — all in a single VLM call per image. For video it samples frames, builds a date consensus across them, and synthesizes a written summary plus structured metadata fields you can review and edit before writing.

---

## Features

### Photos

- **Back-of-photo detection.** When a scan of the back of a print follows the front in filename order, Metadata-AI detects it, reads handwritten dates and comments via the VLM (and optionally Tesseract OCR for an extra pass), and translates non-English captions to English.
- **Single-call image analysis.** One VLM round-trip per photo extracts: date estimate + confidence, time of day, one-sentence scene description, indoor/outdoor setting, whether flash fired, location (if geotagging is on), and 5 keywords.
- **Folder-name as context, validated.** If the folder name carries useful information (`1985 Hawaii`, `Summer 1992`) it's passed to the VLM as a high-confidence hint. Noise like `New Folder (2)`, `Scans_Batch_3`, `untitled`, or `temp` is filtered out and not passed as context.
- **Confidence-gated writes.** Date estimates below your confidence threshold are not written. Instead they're added to a `review.html` report at the end of the run with folder, filename, raw guess, confidence, and any extracted comment.
- **Folder consensus mode.** Optional. After analyzing every photo in a folder, the most common year among high-confidence results is used to override low-confidence dates in the same folder while preserving each photo's individual month, day, and time-of-day.
- **Standards-compliant metadata.** `DateTimeOriginal`, `dc:subject` keywords, GPS, scene caption (`dc:description` / EXIF `ImageDescription`), and `Flash` (proper EXIF `0x9209` structure with `Fired` field) — all written via XMP sidecar then merged into the file with ExifTool.
- **Wide format support.** `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.png`, `.heic`, `.webp`, `.dng`, and raw camera formats (`.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`, `.raw`).
- **Large-scan safe.** Handles 600–1200 DPI scans that exceed 100 MP. Pillow's decompression bomb guard is raised to 500 MP (still catches genuinely corrupt files) and images are downscaled to 2048 px on the long edge before being sent to the VLM.

### Videos

- **Frame-by-frame analysis.** ffmpeg samples one frame every N seconds (default 30) and the VLM produces a one-sentence description, date guess, and confidence per frame.
- **Date consensus across frames.** The most common year among high-confidence frames wins, with a clear plurality-vs-majority distinction in the report when agreement is low.
- **Multi-paragraph summary.** A second VLM call synthesizes the per-frame descriptions into a 2–4 paragraph summary suitable for someone who hasn't seen the video. Truncation is detected and retried with fewer frames.
- **Structured metadata extraction.** A third VLM call distills the summary into `title`, `description`, `keywords`, `location`, `genre`, `artist`, and `date` fields, written into the video container via `ffmpeg -c copy` (no re-encode).
- **Edit-before-write.** Preview every metadata field, edit any of them inline, then choose whether to write to the video file or just save the analysis report.
- **Supported formats.** `.mp4`, `.mov`, `.avi`, `.m4v`, `.mkv`, `.mts`, `.m2ts`, `.wmv`, `.flv`, `.webm`.

### Workflow

- **Resume support.** Long runs are checkpointed file-by-file. If the process is interrupted, you'll be offered the chance to resume from where it left off.
- **Dry-run mode.** `--dry-run` analyzes everything and prints what would be written without modifying any files. Existing checkpoints are ignored in this mode.
- **Skip already-dated photos.** `--skip-dated` skips any file that already has a `DateTimeOriginal` tag — useful for re-runs over a partially completed library.
- **Recursive mode.** Walk an entire archive of subfolders in one run; the review queue is accumulated across all folders into a single `review.html` at the run root.
- **Rate-limited geocoding.** Nominatim lookups respect the 1-second-minimum interval and are cached per location, so a 5,000-photo run won't get you IP-banned.
- **Fully local.** No cloud APIs. The only outbound traffic is to Nominatim, and only if `--geotag` is enabled.

---

## Requirements

- Python 3.8+
- [LM Studio](https://lmstudio.ai) running locally with a vision-capable model loaded
- [ExifTool](https://exiftool.org) for merging XMP into image files
- [ffmpeg](https://ffmpeg.org) and ffprobe for video analysis (included by default in most ffmpeg installs)

Optional:

- [Tesseract](https://github.com/tesseract-ocr/tesseract) + `pytesseract` for an extra OCR pass on the backs of photos
- `rawpy` for opening raw camera formats

On macOS:

```sh
brew install exiftool ffmpeg libheif tesseract
```

---

## Installation

```sh
git clone https://github.com/bcholodenko/Metadata-AI.git
cd Metadata-AI
pip install -r requirements.txt
```

Optional extras:

```sh
pip install pytesseract rawpy
```

Open LM Studio, load a vision-capable model, and start the local server (default: `http://localhost:1234`).

---

## Configuration

The defaults at the top of `metadata-ai.py` are the ones you're most likely to change:

```python
DIRECTORY = "./photos"          # Default fallback if no path is provided
MODEL_ID = "qwen/qwen3.6-27b"   # Must match the model identifier in LM Studio
VLM_MAX_DIMENSION = 2048        # Long-edge pixel cap before sending to VLM
```

Everything else is set per-run via CLI flags or interactive prompts.

---

## Usage

### Photos — fully interactive

```sh
python metadata-ai.py
```

You'll be prompted for the directory, cutoff year, confidence threshold, XMP-only mode, geotagging, recursion, and consensus mode in order.

### Photos — non-interactive

```sh
python metadata-ai.py /path/to/photos --cutoff 1995 --confidence 7 -r --geotag --consensus
```

Any flag you omit will be prompted for.

### Videos

You can point Metadata-AI at a single video file directly:

```sh
python metadata-ai.py /path/to/clip.mov --video-interval 30
```

Or run interactively against a directory and answer "yes" when asked whether to analyze video files. Each video produces a `<name>_summary.txt` report; you'll be shown the metadata preview and asked to confirm before any write to the video file.

### Single image

Pass a single image path and Metadata-AI runs in one-file mode:

```sh
python metadata-ai.py /path/to/scan.tiff
```

### Dry run

```sh
python metadata-ai.py /path/to/photos --dry-run
```

Analyzes everything and prints what it *would* write. No files are modified, no checkpoint is created or read.

### All flags

| Flag | Description | Default |
| --- | --- | --- |
| `directory` (positional) | Path to a photo directory, single image, or video file | Prompted |
| `--cutoff YEAR` | Skip photos dated at or after this year | 2010 |
| `--confidence 1-10` | Minimum confidence to auto-write a date | 7 |
| `--xmp-only` | Write to XMP sidecar files only; skip ExifTool merge | Off |
| `--geotag` | Enable Nominatim geocoding for VLM-identified locations | Off |
| `-r`, `--recursive` | Walk all subfolders | Off |
| `--consensus` | Use folder-consensus year to override low-confidence dates | Off |
| `--dry-run` | Analyze only; modify nothing; ignore checkpoints | Off |
| `--skip-dated` | Skip files that already have `DateTimeOriginal` set | Off |
| `--model ID` | Override the LM Studio model ID for this run | from script |
| `--video PATH` | Run video mode on this file (skips photo prompts) | — |
| `--video-interval SECONDS` | Frame sampling interval for video mode | 30 |
| `--output PATH` | Output `.txt` report path for video mode | `<video>_summary.txt` |

If you scanned the backs of your prints, place them immediately after the front in filename order (e.g. `img001.jpg` front, `img002.jpg` back). Metadata-AI detects and pairs them automatically.

---

## How It Works (Photos)

1. **Back-of-photo check.** The next file in filename order is examined with a single combined VLM prompt: is this the back? If yes, what date is written on it? What other text is on it (translated to English)? When Tesseract is installed and the VLM didn't find a date, an OCR pass tries to recover one from machine-readable text.
2. **IPTC keyword check.** Existing IPTC keywords are scanned for a parseable date — useful when re-running over photos that already have partial metadata.
3. **VLM image analysis.** One call returns date (if still unknown) + confidence, time of day, scene description, indoor/outdoor, flash fired, location (if `--geotag`), and keywords. The folder name is included as context only when it carries real signal.
4. **Date validation.** The estimate is checked against a plausible range (1826–2100 for photos), and the time-of-day estimate is normalized to an hour and applied to `DateTimeOriginal`.
5. **Geotagging.** If enabled, the VLM's location is geocoded via Nominatim. Lookups are rate-limited to 1.1 s and cached.
6. **Write.** Valid dates below the cutoff year are written via XMP sidecar, then merged into the file with `exiftool -overwrite_original -tagsfromfile=...`. The sidecar is removed after a successful merge. If consensus mode is on, writes are deferred until the whole folder is analyzed.

Low-confidence results with no consensus override are appended to a `review.html` report — one row per photo with folder, filename, the raw VLM guess, confidence, and any extracted comment.

---

## What Gets Written Where

| Field | XMP / EXIF target | Notes |
| --- | --- | --- |
| Date | `exif:DateTimeOriginal` | Validated 1826–2100, with estimated hour-of-day |
| Keywords | `dc:subject` | 5 lowercase tags from VLM, time-of-day words filtered out |
| Caption | `dc:description` (EXIF `ImageDescription`) | Scene sentence + back-of-photo comment + setting + raw date string |
| Flash | `exif:Flash` structure with `Fired` field | Proper EXIF tag `0x9209`, readable by Lightroom / Apple Photos / digiKam |
| GPS | `exif:GPSLatitude` / `GPSLongitude` | Only if `--geotag` resolves to a hit |

There is no standard EXIF tag for "indoor vs outdoor" — that lives in the caption alongside the scene description.

---

## Supported Image Formats

| Format | Metadata method |
| --- | --- |
| `.jpg`, `.jpeg` | XMP sidecar merged via ExifTool, sidecar deleted on success |
| `.tiff`, `.tif` | XMP sidecar merged via ExifTool, sidecar deleted on success |
| `.png` | XMP sidecar merged via ExifTool, sidecar deleted on success |
| `.heic`, `.webp` | XMP sidecar merged via ExifTool, sidecar deleted on success |
| `.dng` | XMP sidecar file (kept) |
| `.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.orf`, `.rw2`, `.raw` | Decoded via rawpy for VLM preview, XMP sidecar file (kept) |

When ExifTool is missing or the merge fails, the XMP sidecar is kept beside the file as a fallback — Lightroom, Apple Photos, and digiKam will all read it.

---

## License

MIT
