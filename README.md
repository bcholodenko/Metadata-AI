<img width="600" alt="logo" src="https://github.com/user-attachments/assets/8bcf39fd-93c3-4017-96e0-7d692237b197" />

Metadata-AI tags and dates your scanned photos and videos automatically. Point it at a folder of scans and it figures out roughly when each photo was taken, what's in it, and where it was — then writes that information into the file so any photo app can use it.

It runs entirely on your own computer using a local AI model. Nothing gets uploaded anywhere.

---

## What it does

For each photo:

- If you scanned the back of a print, it reads any handwritten dates and notes (translating them to English if needed).
- It looks at clothing, hairstyles, technology, and setting to estimate when the photo was taken.
- It writes a one-sentence description, five keywords, and tags whether the photo was indoor or outdoor and whether flash was used.
- If you turn on geotagging, it tries to identify the location and look up GPS coordinates — but only when it can see something specific like a landmark or sign, not just a general landscape.

For videos, it samples frames at regular intervals, builds a written summary of what's in the clip, and saves a title, description, keywords, location, and date into the video file.

When the AI isn't sure about a date, it skips writing metadata for that photo and adds it to a review queue instead. At the end of the run you'll be asked whether to walk through those photos one by one in the terminal, with thumbnails of each one shown in a `review.html` file you can keep open in your browser.

---

## Getting started

### What you need

- **Python 3.8 or newer**
- **[LM Studio](https://lmstudio.ai)** — runs the AI model locally
- **[ExifTool](https://exiftool.org)** — writes metadata into your photos
- **ffmpeg** — only if you want to analyze videos

On macOS:

```sh
brew install exiftool ffmpeg libheif
```

### Install

```sh
git clone https://github.com/bcholodenko/Metadata-AI.git
cd Metadata-AI
pip install -r requirements.txt
```

Then open LM Studio, load a vision-capable model, and start its local server.

---

## Using it

The simplest way is to just run it and answer the prompts:

```sh
python metadata-ai.py
```

You'll be asked which folder to process, what cutoff year to use, and a few yes/no questions about what to enable — all with arrow-key selection menus.

If you'd rather skip the prompts:

```sh
python metadata-ai.py /path/to/photos --recursive --geotag --consensus
```

Any flag you omit will be prompted for.

### Useful flags

| Flag | What it does |
| --- | --- |
| `--recursive` (or `-r`) | Process every subfolder, not just the top one |
| `--geotag` | Try to identify locations and add GPS coordinates |
| `--consensus` | Use the most common date in a folder to fix uncertain estimates from the same folder |
| `--review` | Step through previously skipped photos and decide each one |
| `--dry-run` | Show what would be written without changing anything |
| `--skip-dated` | Skip photos that already have a date set |
| `--cutoff YEAR` | Skip photos from this year or later (default: 2010) |
| `--confidence 1-10` | How sure the AI has to be before writing a date (default: 7) |

### Tip: the back of the photo

If you scanned both sides of your prints, save them so the back comes right after the front in filename order — like `img001.jpg` (front) and `img002.jpg` (back). Metadata-AI will pair them automatically and read any handwritten dates or notes.

### Tip: name your folders well

If your folder is called something like `Christmas 1978`, `Summer Vacation`, or `8-79`, the tool will use that as a strong hint. Generic folders like `New Folder (2)` or `Scans_Batch_3` are ignored.

---

## What if it gets something wrong?

- **Interactive review at the end of the run.** When the AI isn't confident about a photo's date, it's skipped rather than given a wrong one. At the end of the run, you'll be offered an interactive review pass: each pending photo is shown with the AI's guess, and you can accept it, enter a different date, or skip the photo permanently. Decisions are saved as you go, so you can quit and resume anytime with `--review`.
- **`review.html`** is a dark-mode visual gallery of every pending photo, with thumbnails embedded inline. Open it in your browser to see what's queued up while you walk through decisions in the terminal. It updates as you go.
- **`--dry-run`** lets you preview a whole run without changing anything.
- **Resume support** is built in. If a long run gets interrupted, you can pick up where you left off.

---

## Supported file types

| Type | Formats | How metadata is saved |
| --- | --- | --- |
| Photos | JPEG, PNG, TIFF, HEIC, WebP | XMP sidecar merged into the file via ExifTool |
| Photos | DNG | XMP sidecar kept next to the file |
| Camera raw | Canon CR2/CR3, Nikon NEF, Sony ARW, Fuji RAF, Olympus ORF, Panasonic RW2 | XMP sidecar kept next to the file (requires `pip install rawpy`) |
| Videos | MP4, MOV, AVI, M4V, MKV, MTS, M2TS, WMV, FLV, WebM | Written directly into the file via ffmpeg |

XMP sidecar files are read by Lightroom, Apple Photos, digiKam, and most photo management apps.

---

## License

GNU General Public License v3 (GPLv3)
