import os
import io
import sys
import base64
import re
import time
import warnings
import json
import shutil
import subprocess
import tempfile
from itertools import groupby
from pathlib import Path
from collections import Counter

# Force line-buffered output so progress prints appear immediately in the terminal
sys.stdout.reconfigure(line_buffering=True)
import logging
from datetime import datetime
from natsort import natsorted
from openai import OpenAI
from PIL import Image
from pillow_heif import register_heif_opener
from iptcinfo3 import IPTCInfo

# Suppress iptcinfo3 logging (it can be very noisy)
logging.getLogger('iptcinfo').setLevel(logging.ERROR)

register_heif_opener()

# Raise Pillow's decompression bomb limit to handle large scanned photos.
# Scanned photos at 600–1200 DPI can easily exceed the default 89MP threshold,
# but we still want a guard against truly absurd / corrupt files. 500 MP covers
# any realistic scan; we additionally cap resolution ourselves before sending to
# the VLM (see get_jpeg_base64).
Image.MAX_IMAGE_PIXELS = 500_000_000

# Maximum long-edge pixel size sent to the VLM. The model doesn't benefit from
# full-resolution images and this avoids unnecessary memory use.
VLM_MAX_DIMENSION = 2048

# Plausible-year ranges used when validating parsed dates.
MIN_PHOTO_YEAR = 1826  # earliest known photograph
MIN_VIDEO_YEAR = 1888  # earliest known motion picture
MAX_YEAR       = 2100

# Configuration
DIRECTORY = "./photos"         # Folder containing your images
MODEL_ID = "qwen/qwen3.6-27b" # Must match the model identifier in LM Studio
CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

EXTENSIONS = ('.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.dng', '.webp', '.cr2', '.cr3', '.nef', '.arw', '.raf', '.orf', '.rw2', '.raw')
RAW_EXTENSIONS = ('.cr2', '.cr3', '.nef', '.arw', '.raf', '.orf', '.rw2', '.raw')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.m4v', '.mkv', '.mts', '.m2ts', '.wmv', '.flv', '.webm')

# ---------------------------------------------------------------------------
# Fuzzy date mapping — maps vague decade/era language to YYYY:MM:DD
# Word boundaries (\b) prevent matches inside larger tokens like "Photo_2024sample".
# ---------------------------------------------------------------------------
FUZZY_DATE_PATTERNS = [
    (r'\bearly\s+(\d{4})s\b',  lambda m: f"{m.group(1)}:01:01 12:00:00"),
    (r'\bmid[- ](\d{4})s\b',   lambda m: f"{str(int(m.group(1))+5)}:01:01 12:00:00"),
    (r'\blate\s+(\d{4})s\b',   lambda m: f"{str(int(m.group(1))+7)}:01:01 12:00:00"),
    (r'\bcirca\s+(\d{4})\b',   lambda m: f"{m.group(1)}:01:01 12:00:00"),
    (r'\bc\.\s*(\d{4})\b',     lambda m: f"{m.group(1)}:01:01 12:00:00"),
    (r'\b(\d{4})s\b',          lambda m: f"{m.group(1)}:01:01 12:00:00"),
]

# ---------------------------------------------------------------------------
# Geocoding rate limiting — Nominatim's TOS requires <= 1 request/second.
# ---------------------------------------------------------------------------
_LAST_GEOCODE_TIME = 0.0
_GEOCODE_MIN_INTERVAL = 1.1  # seconds; small buffer over the 1s minimum
_GEOCODE_CACHE = {}          # location_text -> (lat, lon) or None

def parse_fuzzy_date(text):
    """Try to extract a normalised EXIF date string from vague text. Returns (date_str, raw_text) or (None, None)."""
    if not text:
        return None, None
    # Try unambiguous formats first
    m = re.search(r'(\d{4})[:/-](\d{2})[:/-](\d{2})', text)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)} 12:00:00", None
    # Bare 4-digit year
    m = re.search(r'\b(\d{4})\b', text)
    if m:
        return f"{m.group(1)}:01:01 12:00:00", None
    # Fuzzy decade patterns
    for pattern, formatter in FUZZY_DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return formatter(m), text.strip()
    # Anything else — let the VLM normalize it (handles ranges, short years, etc.)
    normalized = normalize_date_with_vlm(text)
    if normalized:
        return normalized, text.strip()
    return None, None

# ---------------------------------------------------------------------------
# IPTC helper
# ---------------------------------------------------------------------------
def get_iptc_metadata(path):
    """Extracts existing IPTC keywords for date checking."""
    try:
        info = IPTCInfo(path, force=True)
        keywords = [k.decode('utf-8') for k in info['keywords']] if info['keywords'] else []
        return keywords
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Image & API helpers
# ---------------------------------------------------------------------------
def normalize_date_with_vlm(raw_text):
    """Ask the VLM to extract a single year from an ambiguous date string.
    Used as a fallback when parse_fuzzy_date can't resolve cleanly."""
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL_ID,
            messages=[{
                "role": "user",
                "content": (
                    f"Extract the most likely single year from this date string: '{raw_text}'. "
                    "Reply with ONLY a 4-digit year, nothing else. "
                    "If it's a range like '1992-93' or '1992-1993', return the start year."
                )
            }],
            max_tokens=10
        )
        year_str = response.choices[0].message.content.strip()
        m = re.search(r'\b(\d{4})\b', year_str)
        if m:
            return f"{m.group(1)}:01:01 12:00:00"
    except Exception:
        pass
    return None

def get_jpeg_base64(image_path):
    """
    Opens an image, downscales it so the long edge is at most VLM_MAX_DIMENSION,
    converts to JPEG, and returns a base64 string.

    Pillow's decompression bomb guard is raised (Image.MAX_IMAGE_PIXELS = 500M)
    rather than disabled — covers any realistic scan but still catches corrupt
    files. We also explicitly cap the resolution here before encoding to keep
    memory use low and avoid sending unnecessary data to the VLM.

    Raw formats (.cr2, .cr3, .nef, etc.) are decoded via rawpy if available.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
            import numpy as np
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            img = Image.fromarray(rgb)
        except ImportError:
            raise RuntimeError("rawpy is required to open raw files. Install it with: pip install rawpy")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(image_path)
            img.load()  # Force full decode inside the suppression block
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > VLM_MAX_DIMENSION:
        scale = VLM_MAX_DIMENSION / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def ask_vlm(image_path, prompt, max_tokens=600):
    """Sends an image to LM Studio as JPEG regardless of source format.

    A modest max_tokens cap prevents the model from running away on verbose
    completions while leaving plenty of room for the multi-field analysis prompt.
    """
    try:
        base64_image = get_jpeg_base64(image_path)
    except Exception as e:
        print(f"      Could not open image {os.path.basename(image_path)}: {e}")
        return ""
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL_ID,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"      VLM request failed: {e}")
        return ""

def run_tesseract(image_path):
    """Runs Tesseract OCR on an image. Returns extracted text or None if Tesseract is not installed."""
    try:
        import pytesseract
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(image_path)
            img.load()
        return pytesseract.image_to_string(img).strip()
    except ImportError:
        return None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Folder name validation — only inject folder names as VLM context when they
# look meaningful (contain a year or look like place/event names), not when
# they're noise like "New Folder (2)" or "Scans_Batch_3".
# ---------------------------------------------------------------------------
_FOLDER_NOISE_PATTERNS = [
    r'^new\s*folder',
    r'^scans?(\s|_|-)*batch',
    r'^untitled',
    r'^folder\s*\d+$',
    r'^img\s*\d*$',
    r'^batch\s*\d*$',
    r'^temp(orary)?$',
    r'^export(ed)?$',
    r'^unsorted$',
]

def is_meaningful_folder_name(name):
    """Return True if the folder name looks like it carries useful context.

    Heuristic: contains a 4-digit year (very common: '1985 Vacation', 'Summer 1992'),
    OR contains at least 2 alphabetic words and isn't on the noise list.
    """
    if not name or len(name) < 3:
        return False
    lower = name.lower().strip()
    for pattern in _FOLDER_NOISE_PATTERNS:
        if re.search(pattern, lower):
            return False
    # Year present? Almost certainly meaningful.
    if re.search(r'\b(18|19|20)\d{2}\b', name):
        return True
    # At least two alphabetic words of >=3 chars?
    words = [w for w in re.findall(r'[A-Za-z]+', name) if len(w) >= 3]
    return len(words) >= 2

# ---------------------------------------------------------------------------
# Metadata writers
# ---------------------------------------------------------------------------
def apply_metadata(path, date_str, tags=None, comment=None, raw_date=None, gps=None, xmp_only=False, scene=None, setting=None, flash=None):
    ext = os.path.splitext(path)[1].lower()
    if xmp_only or ext in ('.dng',) + RAW_EXTENSIONS:
        _apply_metadata_xmp(path, date_str, tags, comment, raw_date, gps, scene, setting, flash)
    elif ext in ('.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.webp'):
        _apply_metadata_via_exiftool(path, date_str, tags, comment, raw_date, gps, scene, setting, flash)
    else:
        print(f"   ⚠️ Unsupported format for metadata writing: {ext}")

def _apply_metadata_via_exiftool(path, date_str, tags=None, comment=None, raw_date=None, gps=None, scene=None, setting=None, flash=None):
    """Write XMP sidecar then merge it into the file via ExifTool.
    Used for JPEG, TIFF, PNG, HEIC, WebP — anything where we don't want to re-encode pixels.
    The XMP sidecar is deleted after a successful merge; kept as a fallback if ExifTool is missing or fails."""
    xmp_path = os.path.splitext(path)[0] + ".xmp"
    _apply_metadata_xmp(path, date_str, tags, comment, raw_date, gps, scene, setting, flash)

    if shutil.which("exiftool") is None:
        print(f"      ⚠️ ExifTool not found — XMP sidecar kept at {os.path.basename(xmp_path)}")
        return

    try:
        result = subprocess.run(
            ["exiftool", "-overwrite_original", f"-tagsfromfile={xmp_path}", path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            os.remove(xmp_path)
            print(f"   ✅ Success: {os.path.basename(path)} updated via ExifTool.")
        else:
            print(f"      ⚠️ ExifTool merge failed — XMP sidecar kept. Error: {result.stderr.strip()}")
    except Exception as e:
        print(f"      ⚠️ ExifTool error — XMP sidecar kept: {e}")

def _xml_escape(s):
    """Minimal XML escape for text content."""
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))

def _apply_metadata_xmp(path, date_str, tags=None, comment=None, raw_date=None, gps=None, scene=None, setting=None, flash=None):
    try:
        xmp_path = os.path.splitext(path)[0] + ".xmp"
        keywords_xml = ""
        if tags:
            keywords_xml = "".join(
                f"          <rdf:li>{_xml_escape(kw.strip())}</rdf:li>\n" for kw in tags.split(",")
            )

        # Caption / scene description.
        # The VLM's free-text scene sentence belongs in dc:description (the standard
        # XMP caption field, equivalent to EXIF ImageDescription tag 0x010E).
        # Back-of-photo handwritten comments and the raw date string also go here so
        # they're preserved for human review. Indoor/outdoor stays here too — there
        # is no standard EXIF tag for it.
        description_parts = []
        if scene:
            description_parts.append(scene)
        if comment:
            description_parts.append(comment)
        if setting:
            description_parts.append(f"Setting: {setting}")
        if raw_date:
            description_parts.append(f"Raw date: {raw_date}")
        description_xml = ""
        if description_parts:
            joined = _xml_escape(' | '.join(description_parts))
            description_xml = f"      <dc:description><rdf:Alt><rdf:li xml:lang='x-default'>{joined}</rdf:li></rdf:Alt></dc:description>\n"

        # Flash → XMP-exif:Flash structure (flattened to exif:Flash/exif:Fired).
        # Maps to EXIF tag 0x9209 when ExifTool merges into the file.
        flash_xml = ""
        if flash in ('yes', 'no'):
            fired_str = 'True' if flash == 'yes' else 'False'
            flash_xml = (
                f"      <exif:Flash rdf:parseType='Resource'>\n"
                f"        <exif:Fired>{fired_str}</exif:Fired>\n"
                f"      </exif:Flash>\n"
            )

        gps_xml = ""
        if gps:
            lat, lon = gps
            gps_xml = f"      <exif:GPSLatitude>{lat}</exif:GPSLatitude>\n      <exif:GPSLongitude>{lon}</exif:GPSLongitude>\n"
        xmp_content = f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description xmlns:xmp='http://ns.adobe.com/xap/1.0/'
                     xmlns:dc='http://purl.org/dc/elements/1.1/'
                     xmlns:exif='http://ns.adobe.com/exif/1.0/'>
      <exif:DateTimeOriginal>{date_str}</exif:DateTimeOriginal>
{gps_xml}{flash_xml}{description_xml}      <dc:subject>
        <rdf:Bag>
{keywords_xml}        </rdf:Bag>
      </dc:subject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""
        with open(xmp_path, "w", encoding="utf-8") as f:
            f.write(xmp_content)
        print(f"   ✅ Success: {os.path.basename(xmp_path)} written.")
    except Exception as e:
        print(f"   ❌ Metadata Error for {os.path.basename(path)}: {e}")

# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------
def write_review_report(folder, review_queue):
    """Writes an HTML report of low-confidence photos for manual review.

    Called once per run — in recursive mode the caller accumulates the queue
    across all folders and passes the root directory as `folder`.
    """
    if not review_queue:
        return
    report_path = os.path.join(folder, "review.html")
    rows = ""
    for item in review_queue:
        rows += f"""
        <tr>
            <td>{_xml_escape(item.get('folder', ''))}</td>
            <td>{_xml_escape(item['file'])}</td>
            <td>{_xml_escape(item['raw_guess'])}</td>
            <td>{item['confidence']}/10</td>
            <td>{_xml_escape(item.get('comment', '—'))}</td>
        </tr>"""
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Metadata-AI — Review Queue</title>
  <style>
    body {{ font-family: sans-serif; padding: 2em; }}
    h1 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
    th {{ background: #f0f0f0; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>📋 Metadata-AI — Manual Review Queue</h1>
  <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — {len(review_queue)} photo(s) need review.</p>
  <table>
    <thead><tr><th>Folder</th><th>File</th><th>VLM Date Guess</th><th>Confidence</th><th>Comment</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📋 Review report saved: {report_path}")

# ---------------------------------------------------------------------------
# Geotagging
# ---------------------------------------------------------------------------
def geolocate(location_text):
    """Queries Nominatim for GPS coordinates. Returns (lat, lon) or None.

    Rate-limited to <= 1 request/second (Nominatim's TOS) and caches results
    so repeat lookups within a run are free.
    """
    global _LAST_GEOCODE_TIME
    if not location_text:
        return None
    cache_key = location_text.strip().lower()
    if cache_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cache_key]
    try:
        from geopy.geocoders import Nominatim
        # Nominatim asks for a contact-identifying user agent. If you fork this,
        # replace the URL/email with your own.
        geolocator = Nominatim(user_agent="metadata-ai/1.0 (https://github.com/yourname/metadata-ai)")

        # Honour the rate limit
        elapsed = time.monotonic() - _LAST_GEOCODE_TIME
        if elapsed < _GEOCODE_MIN_INTERVAL:
            time.sleep(_GEOCODE_MIN_INTERVAL - elapsed)

        location = geolocator.geocode(location_text, timeout=10)
        _LAST_GEOCODE_TIME = time.monotonic()

        result = (location.latitude, location.longitude) if location else None
        _GEOCODE_CACHE[cache_key] = result
        return result
    except ImportError:
        print("      geopy not installed — skipping GPS tagging.")
    except Exception as e:
        print(f"      Geotagging error: {e}")
    _GEOCODE_CACHE[cache_key] = None
    return None

# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------
def _has_existing_date(path):
    """Returns True if the file already has a DateTimeOriginal tag written by exiftool."""
    if not shutil.which("exiftool"):
        return False
    try:
        result = subprocess.run(
            ["exiftool", "-DateTimeOriginal", "-s3", path],
            capture_output=True, text=True
        )
        return bool(result.stdout.strip())
    except Exception:
        return False

def _checkpoint_path(root_folder):
    return os.path.join(root_folder, ".metadata-ai-progress")

def _load_checkpoint(root_folder):
    path = _checkpoint_path(root_folder)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def _save_checkpoint(root_folder, completed_path):
    with open(_checkpoint_path(root_folder), 'a') as f:
        f.write(completed_path + '\n')

def _clear_checkpoint(root_folder):
    path = _checkpoint_path(root_folder)
    if os.path.exists(path):
        os.remove(path)

def _clean_vlm_field(s):
    """Strip markdown formatting characters from a VLM field value."""
    return re.sub(r'[*_`#]', '', s).strip() if s else s

def _parse_time_of_day(text):
    """Converts a natural language time estimate to an hour (0-23). Returns 12 if unparseable."""
    text = text.lower().strip()
    # Specific time like "3pm", "10am", "14:00", "3:30pm".
    # The minutes group is optional — fixes a bug where bare "3pm" required a colon.
    m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', text)
    if m:
        hour = int(m.group(1))
        ampm = m.group(3)
        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        return min(hour, 23)
    # 24-hour clock like "14:00" or just "14"
    m = re.search(r'\b(\d{1,2}):(\d{2})\b', text)
    if m:
        hour = int(m.group(1))
        return min(hour, 23)
    # Natural language — order matters because we use substring matching:
    # - "early morning" must be checked before "morning"
    # - "late afternoon" / "afternoon" must be checked before "noon" (afternoon contains noon)
    if any(w in text for w in ['dawn', 'sunrise', 'early morning']):
        return 6
    if any(w in text for w in ['late afternoon', 'golden hour']):
        return 17
    if 'afternoon' in text:
        return 14
    if any(w in text for w in ['sunset', 'dusk', 'evening']):
        return 19
    if any(w in text for w in ['midday', 'noon', 'lunch']):
        return 12
    if 'morning' in text:
        return 9
    if 'night' in text:
        return 21
    return 12  # default noon

def _process_folder(folder, files, cutoff_year, confidence_threshold, xmp_only, enable_geo, global_offset=0, global_total=None, folder_consensus=False, root_folder=None, dry_run=False, skip_dated=False, review_queue_accumulator=None):
    """Process one folder of photos.

    review_queue_accumulator: optional shared list. If provided, low-confidence
    items are appended to it instead of (or in addition to) being written to a
    folder-local report. Used by recursive mode to gather everything into a
    single review.html at the run root.
    """
    processed_files = set()
    review_queue = []
    results = []  # collects dicts for deferred consensus write
    completed_paths = _load_checkpoint(root_folder or folder) if not dry_run else set()
    if dry_run and os.path.exists(_checkpoint_path(root_folder or folder)):
        print("   ℹ️  Dry-run: ignoring existing checkpoint, re-analyzing all files.")
    no_date_count = 0
    cutoff_skip_count = 0

    folder_name = os.path.basename(folder)
    folder_name_useful = is_meaningful_folder_name(folder_name)
    print(f"   📁 Folder: {folder_name}" + ("" if folder_name_useful else "  (treated as low-signal — won't be passed to VLM as date/location hint)"))

    print(f"Starting archival of {len(files)} photos in {folder}...")

    for i in range(len(files)):
        current_file = files[i]
        if current_file in processed_files:
            continue

        current_path = os.path.join(folder, current_file)
        found_date = None
        found_comment = None
        raw_date_text = None
        confidence = 10  # default high for back-of-photo and EXIF dates
        gps_coords = None

        pos = global_offset + i + 1
        total_str = str(global_total) if global_total else str(len(files))

        # Resume support — skip files already completed in a previous run
        if current_path in completed_paths:
            print(f"\n[{pos}/{total_str}] Skipping (already processed): {current_file}")
            continue

        # Skip-dated — skip files that already have a DateTimeOriginal tag
        if skip_dated and _has_existing_date(current_path):
            print(f"\n[{pos}/{total_str}] Skipping (already dated): {current_file}")
            continue

        print(f"\n[{pos}/{total_str}] Processing: {current_file}")

        # Step 1: Check if the NEXT photo is the back
        print(f"   1) Checking if next image is back-of-photo...")
        if i + 1 < len(files):
            next_file = files[i+1]
            next_path = os.path.join(folder, next_file)

            # Verify the next file is readable before sending to VLM
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _test = Image.open(next_path)
                    _test.load()
                    _test.close()
                next_readable = True
            except Exception:
                next_readable = False
                print(f"      Could not open {next_file} — skipping back check.")

            # Single VLM call: detect back AND extract date + comment simultaneously
            back_prompt = (
                "Look very carefully at this image. "
                "First determine: is this the BACK (reverse side) of a physical printed photograph? "
                "The back shows blank paper, handwriting, stamps, photo lab printing, or a plain surface — no photographic scene. "
                "Reply in this exact format:\n"
                "IS_BACK: <yes or no>\n"
                "DATE: <any date written on it in YYYY:MM:DD format, or 'circa 1950s', or 'none'>\n"
                "COMMENT: <any other handwritten or printed text excluding dates, translated to English, or 'none'>"
            )
            back_resp = ask_vlm(next_path, back_prompt) if next_readable else ""
            is_back = False
            if back_resp:
                is_back_line = re.search(r'IS_BACK:\s*([^\n]+)', back_resp)
                if is_back_line and is_back_line.group(1).strip().lower().startswith("yes"):
                    is_back = True

            if is_back:
                print(f"      Back confirmed: {next_file}")
                processed_files.add(next_file)

                date_line = re.search(r'DATE:\s*([^\n]+)', back_resp)
                comment_line = re.search(r'COMMENT:\s*([^\n]+)', back_resp)
                raw_date_str = date_line.group(1).strip() if date_line else "none"
                raw_comment = comment_line.group(1).strip() if comment_line else "none"

                # Optionally enrich with Tesseract OCR if available
                ocr_context = run_tesseract(next_path)
                if ocr_context and raw_date_str.lower() == "none":
                    # Tesseract found something the VLM missed — re-ask just for the date
                    print(f"      Tesseract OCR found text — re-checking for date...")
                    ocr_prompt = (
                        f"This is the back of a photo. OCR extracted: {ocr_context}\n"
                        "Extract any date from this text in YYYY:MM:DD format, or 'circa 1950s', etc. "
                        "If no date, return 'none'."
                    )
                    raw_date_str = ask_vlm(next_path, ocr_prompt).strip()

                found_date, raw_date_text = parse_fuzzy_date(raw_date_str)
                if found_date:
                    print(f"      Date from back: {found_date}" + (f" (fuzzy: {raw_date_text})" if raw_date_text else ""))
                else:
                    print(f"      No date on back — falling through to VLM guess.")

                if raw_comment.lower() != "none" and raw_comment:
                    found_comment = raw_comment
                    print(f"      Comment from back: {found_comment}")
                else:
                    print(f"      No comment on back.")
            else:
                print(f"      No back detected.")
        else:
            print(f"      No next image to check.")

        # Step 2: Check IPTC keywords for a date
        if not found_date:
            print(f"   2) Checking IPTC keywords for date...")
            iptc_keywords = get_iptc_metadata(current_path)
            if iptc_keywords:
                for keyword in iptc_keywords:
                    # Skip keywords with no digits — they cannot contain a date
                    if not re.search(r'\d', keyword):
                        continue
                    found_date, raw_date_text = parse_fuzzy_date(keyword)
                    if found_date:
                        try:
                            yr = int(found_date.split(':')[0])
                            if not (MIN_PHOTO_YEAR <= yr <= MAX_YEAR):
                                found_date = None
                                continue
                        except (ValueError, IndexError):
                            found_date = None
                            continue
                        print(f"      Date parsed from IPTC keyword '{keyword}': {found_date}" + (f" (fuzzy: {raw_date_text})" if raw_date_text else ""))
                        break
                if not found_date:
                    print(f"      No date found in IPTC keywords.")
            else:
                print(f"      No IPTC keywords found.")

        # Step 3: Single VLM call — date (if unknown), time, scene, setting, flash,
        # location (if geotagging enabled), and keywords.
        raw_time = None
        vlm_scene = None
        vlm_setting = None
        vlm_flash = None
        tags_resp = None
        geo_resp_inline = None

        print(f"   3) Asking VLM to analyze image...")

        geo_instruction = (
            "LOCATION: <specific city, region, or landmark if clearly identifiable — otherwise 'none'>\n"
        ) if enable_geo else ""

        # Only inject the folder name as VLM context when it looks meaningful —
        # noise like "New Folder (2)" or "Scans_Batch_3" otherwise primes the
        # model with garbage.
        folder_hint_date = (
            f"The folder containing this photo is named '{folder_name}' — treat this as high-confidence information for the date and location. "
            if folder_name_useful else ""
        )
        folder_hint_loc = (
            f"The folder containing this photo is named '{folder_name}' — treat this as high-confidence information for the location.\n"
            if folder_name_useful else ""
        )

        date_instruction = (
            "Analyze the fashion, hairstyles, technology, and setting in this photo. "
            f"{folder_hint_date}"
            "Estimate the date as specifically as possible — could be YYYY:MM, YYYY, a decade like '1970s', or 'circa 1965'. "
            f"The date must be before {cutoff_year}. "
            "Also provide a confidence score from 1-10 for your date estimate.\n"
            "DATE: <your estimate>\n"
            "CONFIDENCE: <score>\n"
        ) if not found_date else folder_hint_loc

        full_prompt = (
            f"{date_instruction}"
            "Also answer the following:\n"
            "TIME: <time of day — e.g. 'morning', 'midday', 'afternoon', 'evening', or '3pm'>\n"
            "SCENE: <one sentence describing the scene>\n"
            "SETTING: <'indoor' or 'outdoor'>\n"
            "FLASH: <'yes' or 'no' — whether flash appears to have fired>\n"
            f"{geo_instruction}"
            "KEYWORDS: <5 descriptive keywords, comma separated>"
        )

        resp = ask_vlm(current_path, full_prompt)
        if not resp:
            print(f"      ⚠️ VLM returned no response — skipping analysis.")
            resp = ""  # ensure regex calls below get an empty string, not None

        # Parse all fields from the single response
        date_line    = re.search(r'DATE:\s*([^\n]+)',       resp)
        conf_line    = re.search(r'CONFIDENCE:\s*(\d+)',    resp)
        time_line    = re.search(r'TIME:\s*([^\n]+)',       resp)
        scene_line   = re.search(r'SCENE:\s*([^\n]+)',      resp)
        setting_line = re.search(r'SETTING:\s*([^\n]+)',    resp)
        flash_line   = re.search(r'FLASH:\s*([^\n]+)',      resp)
        geo_line     = re.search(r'LOCATION:\s*([^\n]+)',   resp) if enable_geo else None
        keywords_line = re.search(r'KEYWORDS:\s*([^\n]+)',  resp)

        if not found_date and date_line:
            raw_guess = date_line.group(1).strip()
            confidence = int(conf_line.group(1)) if conf_line else 5
            found_date, raw_date_text = parse_fuzzy_date(raw_guess)
            if found_date:
                # Validate year, month, and day are in plausible range
                parts = found_date.split(':')
                try:
                    year_val = int(parts[0])
                    month_val = int(parts[1])
                    day_val = int(parts[2].split()[0])
                    if not (MIN_PHOTO_YEAR <= year_val <= MAX_YEAR and 1 <= month_val <= 12 and 1 <= day_val <= 31):
                        print(f"      Invalid date from VLM ('{raw_guess}') — discarding.")
                        found_date = None
                    else:
                        print(f"      Date:       {found_date} (confidence: {confidence}/10)" + (f" — fuzzy: {raw_date_text}" if raw_date_text else ""))
                except (IndexError, ValueError):
                    print(f"      Invalid date format ('{raw_guess}') — discarding.")
                    found_date = None
            else:
                print(f"      VLM could not determine a date. Raw response: '{raw_guess}' (confidence: {confidence}/10)")

        _raw_time_str = _clean_vlm_field(time_line.group(1)) if time_line else None
        # If the VLM returned a long explanation instead of a simple time value, discard it
        raw_time = _raw_time_str if _raw_time_str and len(_raw_time_str) < 40 and not any(
            w in _raw_time_str.lower() for w in [
                "not applicable", "cannot", "studio", "artificial", "controlled",
                "indoor lighting", "specific time", "unable", "n/a", "unknown"
            ]
        ) else None
        vlm_scene   = _clean_vlm_field(scene_line.group(1))        if scene_line   else None
        _raw_setting = _clean_vlm_field(setting_line.group(1)) if setting_line else None
        _raw_flash   = _clean_vlm_field(flash_line.group(1)).lower() if flash_line else None
        # Discard verbose multi-sentence responses — keep only short single-word/phrase answers
        vlm_setting = _raw_setting if _raw_setting and len(_raw_setting) < 30 else None
        vlm_flash   = None
        if _raw_flash:
            if _raw_flash.startswith('yes'):
                vlm_flash = 'yes'
            elif _raw_flash.startswith('no'):
                vlm_flash = 'no'

        # Filter time-of-day words from keywords
        _raw_keywords = _clean_vlm_field(keywords_line.group(1)) if keywords_line else None
        if _raw_keywords:
            _time_words = {"morning", "midday", "noon", "afternoon", "evening",
                           "night", "dawn", "dusk", "sunrise", "sunset", "golden hour"}
            filtered = [k.strip().lower() for k in _raw_keywords.split(',')
                        if k.strip().lower() not in _time_words]
            tags_resp = ', '.join(filtered) if filtered else None
        else:
            tags_resp = None
        geo_resp_inline = _clean_vlm_field(geo_line.group(1))      if geo_line     else None

        if raw_time:    print(f"      Time:       {raw_time}")
        if vlm_scene:   print(f"      Scene:      {vlm_scene}")
        if vlm_setting: print(f"      Setting:    {vlm_setting}")
        if vlm_flash:   print(f"      Flash:      {vlm_flash}")
        if tags_resp:   print(f"      Keywords:   {tags_resp}")

        # Apply estimated time of day to the date string
        time_hour = _parse_time_of_day(raw_time) if raw_time else 12
        if found_date:
            found_date = found_date[:11] + f"{time_hour:02d}:00:00"
            print(f"      Timestamp:  {found_date}" + (f" (~{raw_time})" if raw_time else ""))

        # Step 4: Geotagging — use inline location from VLM if available, else folder hint
        if enable_geo:
            print(f"   4) Checking for location clues...")
            geo_resp = geo_resp_inline or ""
            # Strip parenthetical explanations e.g. "Region 84 (likely Southern California...)"
            geo_resp = re.sub(r'\s*\(.*?\)', '', geo_resp).strip()
            is_valid_location = (
                geo_resp.lower() not in ("", "none")
                and len(geo_resp) < 80
                and geo_resp.lower() != folder_name.lower()
                and not any(phrase in geo_resp.lower() for phrase in [
                    "no identifiable", "no clear", "cannot identify", "unable to",
                    "no location", "no specific", "there are no", "i cannot", "i can't",
                    "unsorted", "folder", "unknown", "region", "likely"
                ])
            )
            if is_valid_location:
                print(f"      Location identified: {geo_resp}")
                gps_coords = geolocate(geo_resp)
                if gps_coords:
                    print(f"      GPS: {gps_coords[0]:.4f}, {gps_coords[1]:.4f}")
                else:
                    # Retry with just the first part of the location (before any comma)
                    simplified = geo_resp.split(',')[0].strip()
                    if simplified and simplified.lower() != geo_resp.lower():
                        print(f"      Retrying with simplified location: '{simplified}'")
                        gps_coords = geolocate(simplified)
                        if gps_coords:
                            print(f"      GPS: {gps_coords[0]:.4f}, {gps_coords[1]:.4f}")
                        else:
                            print(f"      Could not resolve GPS — skipping location tag.")
                    else:
                        print(f"      Could not resolve GPS — skipping location tag.")
            else:
                print(f"      No location identified.")

        # Step 5: Keywords already generated in step 3 — just confirm or skip
        if tags_resp:
            print(f"   5) Keywords from VLM analysis: {tags_resp}")
        else:
            print(f"   5) No keywords returned by VLM.")

        # Step 6: Write metadata (deferred if folder_consensus is on)
        print(f"   6) Writing metadata...")
        if found_date:
            try:
                year = int(found_date[:4])
                if year < cutoff_year:
                    if folder_consensus:
                        # Defer write — append to results for later consensus apply
                        results.append({
                            "file": current_file,
                            "path": current_path,
                            "found_date": found_date,
                            "confidence": confidence,
                            "tags": tags_resp,
                            "comment": found_comment,
                            "raw_date": raw_date_text,
                            "gps": gps_coords,
                            "scene": vlm_scene,
                            "setting": vlm_setting,
                            "flash": vlm_flash,
                        })
                        print(f"      Queued for consensus write.")
                    else:
                        if confidence >= confidence_threshold:
                            if dry_run:
                                print(f"      [DRY RUN] Would write: {found_date} | tags: {tags_resp}")
                            else:
                                apply_metadata(current_path, found_date, tags=tags_resp, comment=found_comment,
                                               raw_date=raw_date_text, gps=gps_coords, xmp_only=xmp_only,
                                               scene=vlm_scene, setting=vlm_setting, flash=vlm_flash)
                                _save_checkpoint(root_folder or folder, current_path)
                        else:
                            print(f"      ⚠️  Low confidence ({confidence}/10) — added to review queue.")
                            review_queue.append({
                                "folder": folder_name,
                                "file": current_file,
                                "raw_guess": raw_date_text or found_date,
                                "confidence": confidence,
                                "comment": found_comment or "—"
                            })
                else:
                    print(f"      ⏭️  Skipping: date {year} is {cutoff_year} or later.")
                    cutoff_skip_count += 1
            except Exception as e:
                print(f"      ❌ Write error: {e}")
        else:
            print(f"      ❌ No date found — skipping.")
            no_date_count += 1

    # Apply folder consensus year if enabled
    if folder_consensus and results:
        # Collect years from high-confidence results only
        high_conf_years = [
            int(r['found_date'][:4])
            for r in results
            if r['confidence'] >= confidence_threshold
        ]
        if high_conf_years:
            consensus_year = Counter(high_conf_years).most_common(1)[0][0]
            print(f"\n   🗳️  Folder consensus year: {consensus_year} "
                  f"(from {len(high_conf_years)} high-confidence result(s))")
        else:
            consensus_year = None
            print(f"\n   ⚠️  No high-confidence results to derive consensus year.")

        for r in results:
            date = r['found_date']
            if consensus_year and r['confidence'] < confidence_threshold:
                # Apply consensus year, keep individual month/day/time
                date = f"{consensus_year}:{date[5:]}"
                print(f"      📅 {r['file']}: low-confidence date overridden to {date} via consensus")
            if r['confidence'] >= confidence_threshold or consensus_year:
                if dry_run:
                    print(f"      [DRY RUN] Would write: {date} | tags: {r['tags']}")
                else:
                    apply_metadata(r['path'], date, tags=r['tags'], comment=r['comment'],
                                   raw_date=r['raw_date'], gps=r['gps'], xmp_only=xmp_only,
                                   scene=r['scene'], setting=r['setting'], flash=r['flash'])
                    _save_checkpoint(root_folder or folder, r['path'])
            else:
                review_queue.append({
                    "folder": folder_name,
                    "file": r['file'],
                    "raw_guess": r['raw_date'] or date,
                    "confidence": r['confidence'],
                    "comment": r['comment'] or "—"
                })

    # Either accumulate into the caller's shared queue (recursive mode), or
    # write a folder-local report (single-folder mode).
    if review_queue_accumulator is not None:
        review_queue_accumulator.extend(review_queue)
    else:
        write_review_report(folder, review_queue)

    # Summary
    total = len(files)
    backs_consumed = len(processed_files)   # back-of-photo files skipped as fronts
    reviewed = len(review_queue)
    written = max(0, total - backs_consumed - reviewed - no_date_count - cutoff_skip_count)

    print(f"\n{'─'*50}")
    print(f"   📊 Folder summary: {total} file(s) scanned")
    print(f"      ✅ {written} written")
    if backs_consumed:
        print(f"      🔄 {backs_consumed} back-of-photo file(s) consumed")
    if cutoff_skip_count:
        print(f"      ⏭️  {cutoff_skip_count} skipped (at or after cutoff year)")
    if reviewed:
        print(f"      📋 {reviewed} added to review queue (low confidence)")
    if no_date_count:
        print(f"      ❌ {no_date_count} skipped (no date found)")
    print(f"{'─'*50}")


def process_archive(folder, cutoff_year=2010, confidence_threshold=7, xmp_only=False, enable_geo=False, recursive=False, folder_consensus=False, dry_run=False, skip_dated=False):
    if not os.path.exists(folder):
        print(f"Directory {folder} not found.")
        return

    if recursive:
        all_paths = []
        for root, _, filenames in os.walk(folder):
            for f in filenames:
                if f.lower().endswith(EXTENSIONS):
                    all_paths.append(os.path.join(root, f))
        all_paths = natsorted(all_paths)
        global_total = len(all_paths)
        print(f"\n📂 Found {global_total} files across all subfolders.")

        # Accumulate review items across every subfolder so the final HTML
        # report contains the full picture, not just the last folder's leftovers.
        accumulated_review_queue = []

        global_offset = 0
        for subfolder, path_iter in groupby(all_paths, key=os.path.dirname):
            subfolder_files = [os.path.basename(p) for p in path_iter]
            print(f"\n📁 Processing folder: {subfolder} ({len(subfolder_files)} files)")
            _process_folder(subfolder, subfolder_files, cutoff_year, confidence_threshold, xmp_only, enable_geo,
                            global_offset=global_offset, global_total=global_total, folder_consensus=folder_consensus,
                            root_folder=folder, dry_run=dry_run, skip_dated=skip_dated,
                            review_queue_accumulator=accumulated_review_queue)
            global_offset += len(subfolder_files)

        # Single combined report at the run root.
        write_review_report(folder, accumulated_review_queue)

        if not dry_run:
            _clear_checkpoint(folder)
        return

    files = natsorted([f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)])
    _process_folder(folder, files, cutoff_year, confidence_threshold, xmp_only, enable_geo, folder_consensus=folder_consensus, dry_run=dry_run, skip_dated=skip_dated)
    if not dry_run:
        _clear_checkpoint(folder)


# ---------------------------------------------------------------------------
# Video analysis prompts
# ---------------------------------------------------------------------------
VIDEO_FRAME_PROMPT = """Analyze this video frame briefly.

Reply in EXACTLY this format — keep DESCRIPTION to one sentence:

DESCRIPTION: <one sentence: who/what/where/action, plus any visible text or logos>
DATE: <YYYY, decade like '1990s', 'circa 1985', or 'unknown'>
CONFIDENCE: <1-10>
"""

VIDEO_SUMMARY_PROMPT = """Below are time-stamped descriptions of frames extracted from a video,
one frame every ~{interval} seconds.

{frame_descriptions}

Write a cohesive, well-structured summary of the video covering: the overall
topic/purpose, key people or subjects, major scenes or segments, any apparent
narrative arc, and notable details. Write 2-4 paragraphs suitable for someone
who hasn't seen the video.
"""

VIDEO_METADATA_PROMPT = """Extract metadata from the video summary below.
Reply with ONLY these lines in EXACTLY this order. One line per field. No extra text.
If a field value is unknown, write the word none.

TITLE: value
DESCRIPTION: value
KEYWORDS: value
LOCATION: value
GENRE: value
ARTIST: value

Rules:
- TITLE: if the video has a clear formal title use it; for home movies or personal footage
  use a descriptive title like "1970s Family Home Movie" or "Summer 1985 Vacation"; otherwise none
- DESCRIPTION: one sentence describing the video
- KEYWORDS: 5-8 lowercase keywords, comma-separated
- LOCATION: specific city or place if clearly identifiable, otherwise none
- GENRE: pick exactly one from this list:
    Home Movie - personal or family footage without a formal production
    Family - family events, gatherings, milestones
    Travel - trips, vacations, sightseeing
    Documentary - structured factual or journalistic content
    Short Film - scripted or produced narrative content
    Sports - athletic events or training
    Event - concerts, ceremonies, parties, graduations
    Nature - wildlife, landscapes, outdoor scenery
    Education - instructional or educational content
    Other - anything that does not fit above
- ARTIST: name of filmmaker if clearly identifiable, otherwise none

Summary:
{summary}
"""

# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def _video_get_duration(video_path):
    """Return video duration in seconds via ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(result.stdout)["format"]["duration"])

def _video_extract_frames(video_path, interval, out_dir):
    """Extract one JPEG frame every interval seconds using a single ffmpeg call.
    Returns list of (timestamp_seconds, image_path) tuples."""
    duration = _video_get_duration(video_path)
    # Use fps filter: 1 frame per interval seconds. Much faster than one call per frame.
    fps = 1.0 / interval
    pattern = os.path.join(out_dir, "frame_%08d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "3",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-500:]}")

    # Reconstruct timestamps from filenames (frame_%08d starts at 1)
    frames = []
    frame_files = sorted(f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg"))
    for idx, fname in enumerate(frame_files):
        ts = idx * interval
        if ts < duration:
            frames.append((ts, os.path.join(out_dir, fname)))
    return frames

def _video_format_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _video_parse_field(text, field):
    """Extract a single field from a structured VLM response."""
    m = re.search(rf'^{field}:\s*(.+)', text, re.MULTILINE | re.IGNORECASE)
    val = m.group(1).strip() if m else ""
    val = re.sub(r'[*_`]', '', val).strip()
    return val if val.lower() not in ("unknown", "none", "") else ""

def _video_parse_year(text):
    """Extract best single year from a date string. Returns int or None."""
    if not text or text.strip().lower() in ("unknown", "none", ""):
        return None
    m = re.search(r'\b(\d{4})\b', text)
    if m:
        yr = int(m.group(1))
        if MIN_VIDEO_YEAR <= yr <= MAX_YEAR:
            return yr
    for pattern, formatter in FUZZY_DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            # FUZZY_DATE_PATTERNS return EXIF strings — extract just the year
            yr_str = formatter(m)[:4]
            try:
                yr = int(yr_str)
                if MIN_VIDEO_YEAR <= yr <= MAX_YEAR:
                    return yr
            except ValueError:
                pass
    return None

def _video_consensus_year(year_conf_pairs, threshold):
    eligible = [(yr, c) for yr, c in year_conf_pairs if yr and c >= threshold]
    if not eligible:
        return None, 0, 0, False
    counts = Counter(yr for yr, _ in eligible)
    best, best_votes = counts.most_common(1)[0]
    majority = best_votes > len(eligible) / 2
    return best, best_votes, len(eligible), majority

def _video_analyze_frame(ts, image_path, index, total, video_name=""):
    """Analyze a single frame. Returns dict with description, date_raw, year, confidence."""
    print(f"  Frame {index}/{total}  [{_video_format_ts(ts)}] ...", end=" ")
    _looks_technical = bool(re.search(
        r'(fps|mbps|kbps|\d+x\d+|bitrate|codec|h264|h265|hevc|avc)',
        video_name, re.IGNORECASE))
    context_hint = (f"\nContext: this frame is from a video file named '{video_name}'. "
                    "Treat this as high-confidence date and location information.")\
                   if video_name and not _looks_technical else ""
    resp = ask_vlm(image_path, VIDEO_FRAME_PROMPT + context_hint)
    if not resp:
        print("no response")
        return {"description": "", "date_raw": "", "year": None, "confidence": 0}
    description = _video_parse_field(resp, "DESCRIPTION")
    date_raw    = _video_parse_field(resp, "DATE") or "unknown"
    conf_str    = _video_parse_field(resp, "CONFIDENCE")
    try:
        confidence = max(1, min(10, int(re.search(r'\d+', conf_str).group())))
    except Exception:
        confidence = 5
    year = _video_parse_year(date_raw)
    flag = "✓" if confidence >= 6 and year is not None else "~"
    date_display = f"{date_raw} → {year}" if year else date_raw
    print(f"done  [{flag} {date_display}, {confidence}/10]")
    return {"description": description, "date_raw": date_raw, "year": year, "confidence": confidence}

def _video_synthesize_summary(frame_analyses, interval):
    """Ask the VLM to synthesize all frame descriptions into a summary."""
    all_descriptions = [
        f"[{_video_format_ts(ts)}]\n{d['description']}"
        for ts, d in frame_analyses if d.get("description")
    ]
    # Cap context to ~6000 tokens (≈24000 chars) for the descriptions block.
    # With short one-sentence frame descriptions this fits ~300-400 frames comfortably.
    # If over budget, evenly downsample until it fits.
    MAX_CHARS = 24000
    selected = all_descriptions
    while len(selected) > 1 and sum(len(s) for s in selected) > MAX_CHARS:
        selected = selected[::2]
    if len(selected) < len(all_descriptions):
        print(f"      (Summarizing {len(selected)} of {len(all_descriptions)} frames to fit context window)")
    frame_descriptions = "\n\n".join(selected)
    prompt = VIDEO_SUMMARY_PROMPT.format(interval=interval, frame_descriptions=frame_descriptions)
    print("Synthesizing summary ...", end=" ")
    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL_ID,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        result = resp.choices[0].message.content.strip()

        # Detect truncation — if response ends mid-sentence, retry with fewer frames
        finish_reason = getattr(resp.choices[0], 'finish_reason', None)
        # Detect truncation either via finish_reason or by checking if response ends mid-sentence
        ends_mid_sentence = result and not result.rstrip().endswith(('.', '!', '?', '"', ')', ']'))
        if (finish_reason == "length" or ends_mid_sentence) and len(selected) > 1:
            print(f"truncated — retrying with {len(selected)//2} frames ...", end=" ")
            selected = selected[::2]
            frame_descriptions = "\n\n".join(selected)
            prompt2 = VIDEO_SUMMARY_PROMPT.format(interval=interval, frame_descriptions=frame_descriptions)
            resp2 = CLIENT.chat.completions.create(
                model=MODEL_ID,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt2}]
            )
            result = resp2.choices[0].message.content.strip()
            retry_finish = getattr(resp2.choices[0], 'finish_reason', None)
            if retry_finish == "length":
                print("(retry also truncated — summary may still be incomplete)", end=" ")

        print("done")
        return result
    except Exception as e:
        print(f"failed: {e}")
        return "(Summary generation failed.)"

def _video_extract_metadata(summary, consensus_yr):
    """Ask the VLM to extract structured metadata from the summary."""
    print("Extracting metadata fields ...", end=" ")
    prompt = VIDEO_METADATA_PROMPT.format(summary=summary)
    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content.strip()
        print("done")
    except Exception as e:
        print(f"failed: {e}")
        text = ""
    return {
        "title":       _video_parse_field(text, "TITLE"),
        "description": _video_parse_field(text, "DESCRIPTION"),
        "keywords":    _video_parse_field(text, "KEYWORDS"),
        "location":    _video_parse_field(text, "LOCATION"),
        "genre":       _video_parse_field(text, "GENRE"),
        "artist":      _video_parse_field(text, "ARTIST"),
        "date":        str(consensus_yr) if consensus_yr else "",
    }

def _video_write_metadata(video_path, metadata, summary):
    """Write metadata into video container via ffmpeg stream copy. Returns True on success."""
    tmp_path = video_path + ".tmp" + Path(video_path).suffix
    meta_args = []
    for key, val in {
        "title":       metadata.get("title"),
        "description": metadata.get("description"),
        "comment":     summary,
        "keywords":    metadata.get("keywords"),
        "date":        metadata.get("date"),
        "location":    metadata.get("location"),
        "genre":       metadata.get("genre"),
        "artist":      metadata.get("artist"),
    }.items():
        if val:
            meta_args += ["-metadata", f"{key}={val}"]

    cmd = ["ffmpeg", "-y", "-i", video_path, "-c", "copy", *meta_args, tmp_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ⚠️  ffmpeg error: {result.stderr[-300:]}")
            if os.path.exists(tmp_path): os.remove(tmp_path)
            return False
        os.replace(tmp_path, video_path)

        return True
    except Exception as e:
        print(f"   ⚠️  ffmpeg exception: {e}")
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return False

def _video_build_report(video_path, interval, frame_analyses, summary, consensus_yr):
    lines = [
        "=" * 72, "VIDEO ANALYSIS REPORT", "=" * 72,
        f"File    : {os.path.basename(video_path)}",
        f"Model   : {MODEL_ID}",
        f"Interval: every {interval} seconds",
        f"Frames  : {len(frame_analyses)}",
        f"Date    : {consensus_yr if consensus_yr else 'unknown'}",
        "", "SUMMARY", "-" * 72, summary, "",
        "FRAME-BY-FRAME ANALYSIS", "-" * 72,
    ]
    for ts, d in frame_analyses:
        desc = d.get("description") or "(no description)"
        yr   = d.get("year")
        conf = d.get("confidence", 0)
        raw  = d.get("date_raw", "")
        note = f"  [date: {raw} → {yr}, confidence: {conf}/10]" if yr else f"  [date: {raw}, confidence: {conf}/10]"
        lines.append(f"\n[{_video_format_ts(ts)}]{note}")
        lines.append(desc)
    lines += ["", "=" * 72]
    return "\n".join(lines)

def process_video(video_path, interval=30, output_path=None):
    """Main video analysis pipeline. Prompts user for options interactively."""
    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        return

    # Output path
    default_output = str(Path(video_path).parent / (Path(video_path).stem + "_summary.txt"))
    if output_path is None:
        output_input = input(f"Output summary file [{default_output}]: ").strip()
        output_path = output_input or default_output

    print(f"\n📹  Video   : {video_path}")
    print(f"🤖  Model   : {MODEL_ID}")
    print(f"⏱️  Interval: every {interval}s")
    print(f"💾  Output  : {output_path}\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Extracting frames with ffmpeg...")
        try:
            frames = _video_extract_frames(video_path, interval, tmp_dir)
        except Exception as e:
            print(f"   ❌ Frame extraction failed: {e}")
            return
        print(f"  → {len(frames)} frame(s) extracted\n")

        print("Analyzing frames...")
        frame_analyses = []
        video_stem = Path(video_path).stem
        for i, (ts, img_path) in enumerate(frames, 1):
            data = _video_analyze_frame(ts, img_path, i, len(frames), video_name=video_stem)
            frame_analyses.append((ts, data))
        print()

        year_conf_pairs = [(d["year"], d["confidence"]) for _, d in frame_analyses]
        cons_year, votes, eligible, majority = _video_consensus_year(year_conf_pairs, threshold=6)

        print("─" * 50)
        print("DATE CONSENSUS")
        print("─" * 50)
        if cons_year:
            majority_str = "" if majority else " ⚠️  (plurality only — low agreement)"
            print(f"  Consensus year : {cons_year}{majority_str}")
            print(f"  Votes          : {votes}/{eligible} high-confidence frame(s)")
        else:
            print("  No consensus date — not enough confident frame estimates.")

        print("\n  Per-frame breakdown:")
        for ts, d in frame_analyses:
            flag   = "✓" if d["confidence"] >= 6 and d["year"] is not None else "~"
            yr_str = str(d["year"]) if d["year"] else "?"
            print(f"    [{_video_format_ts(ts)}]  {flag} {d['date_raw']:<20} → {yr_str:<6}  ({d['confidence']}/10)")
        print("─" * 50)
        print()

        summary  = _video_synthesize_summary(frame_analyses, interval)
        print()
        metadata = _video_extract_metadata(summary, cons_year)
        print()

    print("=" * 72)
    print("METADATA PREVIEW")
    print("=" * 72)
    print(f"  Title      : {metadata['title']      or '(none)'}")
    print(f"  Description: {metadata['description'] or '(none)'}")
    print(f"  Keywords   : {metadata['keywords']    or '(none)'}")
    print(f"  Date       : {metadata['date']        or '(none)'}")
    print(f"  Location   : {metadata['location']    or '(none)'}")
    print(f"  Genre      : {metadata['genre'] or '(none)'}")
    print(f"  Artist     : {metadata['artist']      or '(none)'}")
    summary_preview = summary[:500].replace("\n", " ") + ("..." if len(summary) > 500 else "")
    print(f"  Summary    :")
    # Word-wrap the summary preview at 70 chars
    import textwrap
    for line in textwrap.wrap(summary_preview, width=68):
        print(f"    {line}")
    print("=" * 72)

    # Offer to edit each field before writing
    edit = input("\nEdit metadata fields before writing? [y/N]: ").strip().lower() == "y"
    if edit:
        print("  (Press Enter to keep current value)")
        for field in ["title", "description", "keywords", "date", "location", "genre", "artist"]:
            current = metadata.get(field) or ""
            new_val = input(f"  {field.capitalize()} [{current}]: ").strip()
            if new_val:
                metadata[field] = new_val

        # Offer to edit summary too
        print(f"\n  Summary ({len(summary)} chars — shown first 200):")
        print(f"  {summary[:200].replace(chr(10), ' ')}...")
        new_summary = input("  Replace summary? (paste new text, or Enter to keep): ").strip()
        if new_summary:
            summary = new_summary

    write_meta = input("\nWrite metadata to video file? [y/N]: ").strip().lower() == "y"
    if write_meta:
        print(f"\nWriting metadata to {os.path.basename(video_path)} ...", end=" ")
        ok = _video_write_metadata(video_path, metadata, summary)
        if ok:
            print("✅  Metadata written successfully.")
        else:
            print("❌  Metadata write failed — original file unchanged.")
    else:
        print("   Skipping metadata write.")

    report = _video_build_report(video_path, interval, frame_analyses, summary, cons_year)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅  Report saved to: {output_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Metadata-AI — automatically tag and date scanned photos using a local VLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python metadata-ai.py                          # fully interactive\n"
            "  python metadata-ai.py /path/to/directory       # prompts for remaining options\n"
            "  python metadata-ai.py /path/to/directory -r --geotag --consensus\n"
            "  python metadata-ai.py /path/to/directory --cutoff 1995 --confidence 6 --xmp-only"
        )
    )
    parser.add_argument("directory", nargs="?", default=None,
                        help="Path to directory containing photos or videos (prompted if omitted)")
    parser.add_argument("--cutoff", type=int, default=None, metavar="YEAR",
                        help="Skip photos dated from this year or later (default: 2010)")
    parser.add_argument("--confidence", type=int, default=None, metavar="1-10",
                        help="Confidence threshold for auto-write (default: 7)")
    parser.add_argument("--xmp-only", action="store_true", default=None,
                        help="Write metadata to XMP sidecar files only")
    parser.add_argument("--geotag", action="store_true", default=None,
                        help="Enable geotagging via Nominatim")
    parser.add_argument("-r", "--recursive", action="store_true", default=None,
                        help="Recursively process all subfolders")
    parser.add_argument("--consensus", action="store_true", default=None,
                        help="Use folder consensus year to correct low-confidence date estimates")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Analyze photos and print what would be written without modifying any files")
    parser.add_argument("--skip-dated", action="store_true", default=False,
                        help="Skip photos that already have a DateTimeOriginal tag written")
    parser.add_argument("--model", type=str, default=None,
                        help=f"LM Studio model ID to use (default: {MODEL_ID})")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file to analyze (skips photo processing)")
    parser.add_argument("--video-interval", type=int, default=None, metavar="SECONDS",
                        help="Seconds between frames for video analysis (default: 30)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output summary .txt path for video analysis (default: <video>_summary.txt)")

    args = parser.parse_args()

    # Apply --model override globally so all VLM helpers see it.
    if args.model:
        MODEL_ID = args.model
        print(f"Using model: {MODEL_ID}")

    # Directory or file — prompt if not provided
    if args.directory:
        input_path = re.sub(r'\\(.)', r'\1', args.directory.strip())
    else:
        raw = input(f"Enter directory or file path [{DIRECTORY}]: ").strip()
        input_path = re.sub(r'\\(.)', r'\1', raw) or DIRECTORY

    # If a file path was given, route directly to the right processor
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            if args.video_interval is not None:
                interval = args.video_interval
            else:
                interval_input = input("Seconds between frames [30]: ").strip()
                try:
                    interval = int(interval_input) if interval_input else 30
                except ValueError:
                    interval = 30
            output_path = getattr(args, 'output', None)
            process_video(input_path, interval, output_path=output_path)
        elif ext in EXTENSIONS:
            # Single image — create a minimal one-file "folder" run
            folder = os.path.dirname(input_path) or "."
            filename = os.path.basename(input_path)
            cutoff_year = args.cutoff if args.cutoff is not None else 2010
            confidence_threshold = args.confidence if args.confidence is not None else 7
            xmp_only = args.xmp_only or False
            enable_geo = args.geotag or False
            _process_folder(folder, [filename], cutoff_year, confidence_threshold,
                            xmp_only, enable_geo,
                            folder_consensus=args.consensus or False,
                            dry_run=args.dry_run,
                            skip_dated=args.skip_dated)
        else:
            print(f"Unsupported file type: {ext}")
        sys.exit(0)

    directory = input_path

    # ── Video mode — check early so no photo prompts are shown ───────────────
    if args.video:
        # Fully CLI-driven: resolve output path from --output if provided
        video_path = re.sub(r'\\(.)', r'\1', args.video.strip())
        interval = args.video_interval if args.video_interval is not None else 30
        output_path = getattr(args, 'output', None)
        process_video(video_path, interval, output_path=output_path)
        sys.exit(0)

    # ── Photo or video mode selection ─────────────────────────────────────────
    analyze_video = input("Analyze video files in this directory? [y/N]: ").strip().lower() == "y"
    if analyze_video:
        video_files = natsorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ])
        if not video_files:
            print("   No video files found — switching to photo mode.\n")
        else:
            interval_input = input("Seconds between frames [30]: ").strip()
            try:
                video_interval = int(interval_input) if interval_input else 30
            except ValueError:
                video_interval = 30
            if not args.model:
                model_input = input(f"LM Studio model [{MODEL_ID}]: ").strip()
                if model_input:
                    MODEL_ID = model_input
                    print(f"Using model: {MODEL_ID}")
            print(f"\n   Found {len(video_files)} video file(s).")
            for vf in video_files:
                print(f"\n{'─'*50}")
                print(f"📹 {vf}")
                print(f"{'─'*50}")
                process_video(os.path.join(directory, vf), video_interval)
            sys.exit(0)

    # ── Photo mode ───────────────────────────────────────────────────────────
    if args.cutoff is not None:
        cutoff_year = args.cutoff
    else:
        cutoff_input = input("Skip photos dated from which year or later? [2010]: ").strip()
        try:
            cutoff_year = int(cutoff_input) if cutoff_input else 2010
        except ValueError:
            print("Invalid year, defaulting to 2010.")
            cutoff_year = 2010

    if args.confidence is not None:
        confidence_threshold = args.confidence
    else:
        conf_input = input("Confidence threshold for auto-write (1-10) [7]: ").strip()
        try:
            confidence_threshold = int(conf_input) if conf_input else 7
        except ValueError:
            confidence_threshold = 7

    if args.xmp_only is not None:
        xmp_only = args.xmp_only
    else:
        xmp_only = input("Write metadata to XMP sidecar files only? [y/N]: ").strip().lower() == "y"

    if args.geotag is not None:
        enable_geo = args.geotag
    else:
        enable_geo = input("Enable geotagging? [y/N]: ").strip().lower() == "y"

    if args.recursive is not None:
        recursive = args.recursive
    else:
        recursive = input("Recursively process subfolders? [y/N]: ").strip().lower() == "y"

    if args.consensus is not None:
        folder_consensus = args.consensus
    else:
        folder_consensus = input("Average dates in each folder using consensus year? [y/N]: ").strip().lower() == "y"

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE — no files will be modified.\n")

    # Check for an existing progress file and offer to resume.
    # Skipped in dry-run mode — dry runs ignore checkpoints (set inside _process_folder).
    if not args.dry_run:
        progress_file = _checkpoint_path(directory)
        if os.path.exists(progress_file):
            with open(progress_file) as pf:
                completed_count = sum(1 for line in pf if line.strip())
            resume = input(f"\n⏸️  Found a previous session with {completed_count} completed file(s). Resume? [Y/n]: ").strip().lower()
            if resume == "n":
                _clear_checkpoint(directory)
                print("   Starting fresh.\n")
            else:
                print("   Resuming previous session.\n")

    process_archive(directory, cutoff_year, confidence_threshold, xmp_only, enable_geo, recursive, folder_consensus, dry_run=args.dry_run, skip_dated=args.skip_dated)
