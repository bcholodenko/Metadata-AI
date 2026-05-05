import os
import io
import sys
import base64
import re
import warnings

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
# Scanned photos at 600–1200 DPI can easily exceed the default 89MP threshold.
# Setting to None disables the limit entirely; we cap resolution ourselves before
# sending to the VLM (see get_jpeg_base64), so there is no memory blowout risk.
Image.MAX_IMAGE_PIXELS = None

# Maximum long-edge pixel size sent to the VLM. The model doesn't benefit from
# full-resolution images and this avoids unnecessary memory use.
VLM_MAX_DIMENSION = 2048

# Configuration
DIRECTORY = "./photos"         # Folder containing your images
MODEL_ID = "qwen/qwen3.6-27b" # Must match the model identifier in LM Studio
CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

EXTENSIONS = ('.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.dng', '.webp', '.cr2', '.cr3', '.nef', '.arw', '.raf', '.orf', '.rw2', '.raw')
RAW_EXTENSIONS = ('.cr2', '.cr3', '.nef', '.arw', '.raf', '.orf', '.rw2', '.raw')

# ---------------------------------------------------------------------------
# Fuzzy date mapping — maps vague decade/era language to YYYY:MM:DD
# ---------------------------------------------------------------------------
FUZZY_DATE_PATTERNS = [
    (r'early\s+(\d{4})s',  lambda m: f"{m.group(1)}:01:01 12:00:00"),
    (r'mid[- ](\d{4})s',   lambda m: f"{str(int(m.group(1))+5)}:01:01 12:00:00"),
    (r'late\s+(\d{4})s',   lambda m: f"{str(int(m.group(1))+7)}:01:01 12:00:00"),
    (r'circa\s+(\d{4})',   lambda m: f"{m.group(1)}:01:01 12:00:00"),
    (r'c\.\s*(\d{4})',     lambda m: f"{m.group(1)}:01:01 12:00:00"),
    (r'(\d{4})s',          lambda m: f"{m.group(1)}:01:01 12:00:00"),
]

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

    Large scanned photos (600+ DPI) can easily exceed 100 MP. Pillow's decompression
    bomb guard is disabled at module level (Image.MAX_IMAGE_PIXELS = None), so we
    explicitly cap the resolution here before encoding — keeping memory use low and
    avoiding unnecessary data being sent to the VLM.

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

def ask_vlm(image_path, prompt):
    """Sends an image to LM Studio as JPEG regardless of source format."""
    try:
        base64_image = get_jpeg_base64(image_path)
    except Exception as e:
        print(f"      Could not open image {os.path.basename(image_path)}: {e}")
        return ""
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL_ID,
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
# Metadata writers
# ---------------------------------------------------------------------------
def apply_metadata(path, date_str, tags=None, comment=None, raw_date=None, gps=None, xmp_only=False, scene=None, setting=None, flash=None):
    ext = os.path.splitext(path)[1].lower()
    if xmp_only or ext in ('.dng',) + RAW_EXTENSIONS:
        _apply_metadata_xmp(path, date_str, tags, comment, raw_date, gps, scene, setting, flash)
    elif ext in ('.jpg', '.jpeg'):
        _apply_metadata_jpeg(path, date_str, tags, comment, raw_date, gps, scene, setting, flash)
    elif ext in ('.tiff', '.tif'):
        _apply_metadata_tiff(path, date_str, tags, comment, raw_date, gps, scene, setting, flash)
    elif ext in ('.png', '.heic', '.webp'):
        _apply_metadata_png(path, date_str, tags, comment, raw_date, gps, scene, setting, flash)
    else:
        print(f"   ⚠️ Unsupported format for metadata writing: {ext}")

def _apply_metadata_jpeg(path, date_str, tags=None, comment=None, raw_date=None, gps=None, scene=None, setting=None, flash=None):
    # XMP+ExifTool approach for consistency and to avoid any
    import shutil, subprocess
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

def _apply_metadata_tiff(path, date_str, tags=None, comment=None, raw_date=None, gps=None, scene=None, setting=None, flash=None):
    # Write XMP sidecar first, then use ExifTool to merge it safely into the TIFF's
    # EXIF without re-encoding any pixel data. If ExifTool is not available, the XMP
    # sidecar is kept as a fallback.
    import shutil, subprocess
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


def _apply_metadata_png(path, date_str, tags=None, comment=None, raw_date=None, gps=None, scene=None, setting=None, flash=None):
    # Same XMP+ExifTool approach as TIFF — avoids re-encoding pixel data.
    import shutil, subprocess
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

def _apply_metadata_xmp(path, date_str, tags=None, comment=None, raw_date=None, gps=None, scene=None, setting=None, flash=None):
    try:
        xmp_path = os.path.splitext(path)[0] + ".xmp"
        keywords_xml = ""
        if tags:
            keywords_xml = "".join(
                f"          <rdf:li>{kw.strip()}</rdf:li>\n" for kw in tags.split(",")
            )
        comment_parts = []
        if comment:
            comment_parts.append(comment)
        if raw_date:
            comment_parts.append(f"Raw date: {raw_date}")
        comment_xml = ""
        if comment_parts:
            comment_xml = f"      <dc:description><rdf:Alt><rdf:li xml:lang='x-default'>{' | '.join(comment_parts)}</rdf:li></rdf:Alt></dc:description>\n"
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
{gps_xml}{comment_xml}      <dc:subject>
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
        print(f"   💾 Metadata Error for {os.path.basename(path)}: {e}")

# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------
def write_review_report(folder, review_queue, root_folder=None):
    """Writes an HTML report of low-confidence photos for manual review.
    In recursive mode, root_folder is passed so all folders share one report."""
    if not review_queue:
        return
    report_path = os.path.join(root_folder or folder, "review.html")
    rows = ""
    for item in review_queue:
        rows += f"""
        <tr>
            <td>{item['file']}</td>
            <td>{item['raw_guess']}</td>
            <td>{item['confidence']}/10</td>
            <td>{item.get('comment', '—')}</td>
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
    <thead><tr><th>File</th><th>VLM Date Guess</th><th>Confidence</th><th>Comment</th></tr></thead>
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
    """Queries Nominatim for GPS coordinates. Returns (lat, lon) or None."""
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="metadata-ai")
        location = geolocator.geocode(location_text, timeout=10)
        if location:
            return (location.latitude, location.longitude)
    except ImportError:
        print("      geopy not installed — skipping GPS tagging.")
    except Exception as e:
        print(f"      Geotagging error: {e}")
    return None

# ---------------------------------------------------------------------------
# Main archival loop
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------
def _has_existing_date(path):
    """Returns True if the file already has a DateTimeOriginal tag written by exiftool."""
    import subprocess, shutil
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

def _parse_time_of_day(text):
    """Converts a natural language time estimate to an hour (0-23). Returns 12 if unparseable."""
    text = text.lower().strip()
    # Specific time like "3pm", "10am", "14:00"
    m = re.search(r'(\d{1,2})(?::(\d{2}))\s*(am|pm)?', text)
    if m:
        hour = int(m.group(1))
        ampm = m.group(3)
        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        return min(hour, 23)
    m = re.search(r'(\d{1,2})\s*(am|pm)', text)
    if m:
        hour = int(m.group(1))
        ampm = m.group(2)
        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        return min(hour, 23)
    # Natural language
    if any(w in text for w in ['dawn', 'sunrise', 'early morning']):
        return 6
    if 'morning' in text:
        return 9
    if any(w in text for w in ['midday', 'noon', 'lunch']):
        return 12
    if 'afternoon' in text:
        return 14
    if any(w in text for w in ['late afternoon', 'golden hour']):
        return 17
    if any(w in text for w in ['sunset', 'dusk', 'evening']):
        return 19
    if 'night' in text:
        return 21
    return 12  # default noon

def _process_folder(folder, files, cutoff_year, confidence_threshold, xmp_only, enable_geo, global_offset=0, global_total=None, folder_consensus=False, root_folder=None, dry_run=False, skip_dated=False):
    processed_files = set()
    review_queue = []
    results = []  # collects (path, found_date, confidence, tags, comment, raw_date, gps, scene, setting, flash)
    completed_paths = _load_checkpoint(root_folder or folder)
    no_date_count = 0
    cutoff_skip_count = 0

    # Pass folder name directly to VLM as context
    folder_name = os.path.basename(folder)
    print(f"   📁 Folder: {folder_name}")

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
        back_confirmed = False
        print(f"   1) Checking if next image is back-of-photo...")
        if i + 1 < len(files):
            next_file = files[i+1]
            next_path = os.path.join(folder, next_file)

            # Verify the next file is readable before sending to VLM
            try:
                import warnings
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
                back_confirmed = True
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
                            if not (1826 <= yr <= 2100):
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
        # location (if geotagging enabled), and keywords. Combining into one prompt
        # eliminates multiple round-trips to the model.
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

        date_instruction = (
            "Analyze the fashion, hairstyles, technology, and setting in this photo. "
            f"The folder containing this photo is named '{folder_name}' — treat this as high-confidence information for the date and location. "
            "Estimate the date as specifically as possible — could be YYYY:MM, YYYY, a decade like '1970s', or 'circa 1965'. "
            f"The date must be before {cutoff_year}. "
            "Also provide a confidence score from 1-10 for your date estimate.\n"
            "DATE: <your estimate>\n"
            "CONFIDENCE: <score>\n"
        ) if not found_date else (
            f"The folder containing this photo is named '{folder_name}' — treat this as high-confidence information for the location.\n"
        )

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
                    if not (1826 <= year_val <= 2100 and 1 <= month_val <= 12 and 1 <= day_val <= 31):
                        print(f"      Invalid date from VLM ('{raw_guess}') — discarding.")
                        found_date = None
                    else:
                        print(f"      Date:       {found_date} (confidence: {confidence}/10)" + (f" — fuzzy: {raw_date_text}" if raw_date_text else ""))
                except (IndexError, ValueError):
                    print(f"      Invalid date format ('{raw_guess}') — discarding.")
                    found_date = None
            else:
                print(f"      VLM could not determine a date. Raw response: '{raw_guess}' (confidence: {confidence}/10)")

        def clean(s):
            """Strip markdown formatting characters from VLM field values."""
            return re.sub(r'[*_`#]', '', s).strip() if s else s

        _raw_time_str = clean(time_line.group(1)) if time_line else None
        # If the VLM returned a long explanation instead of a simple time value, discard it
        raw_time = _raw_time_str if _raw_time_str and len(_raw_time_str) < 40 and not any(
            w in _raw_time_str.lower() for w in [
                "not applicable", "cannot", "studio", "artificial", "controlled",
                "indoor lighting", "specific time", "unable", "n/a", "unknown"
            ]
        ) else None
        vlm_scene   = clean(scene_line.group(1))        if scene_line   else None
        _raw_setting = clean(setting_line.group(1)) if setting_line else None
        _raw_flash   = clean(flash_line.group(1)).lower() if flash_line else None
        # Discard verbose multi-sentence responses — keep only short single-word/phrase answers
        vlm_setting = _raw_setting if _raw_setting and len(_raw_setting) < 30 else None
        vlm_flash   = None
        if _raw_flash:
            if _raw_flash.startswith('yes'):
                vlm_flash = 'yes'
            elif _raw_flash.startswith('no'):
                vlm_flash = 'no'

        # Filter time-of-day words from keywords
        _raw_keywords = clean(keywords_line.group(1)) if keywords_line else None
        if _raw_keywords:
            _time_words = {"morning", "midday", "noon", "afternoon", "evening",
                           "night", "dawn", "dusk", "sunrise", "sunset", "golden hour"}
            filtered = [k.strip().lower() for k in _raw_keywords.split(',')
                        if k.strip().lower() not in _time_words]
            tags_resp = ', '.join(filtered) if filtered else None
        else:
            tags_resp = None
        geo_resp_inline = clean(geo_line.group(1))      if geo_line     else None

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
                    if not folder_consensus:
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
                                "file": current_file,
                                "raw_guess": raw_date_text or found_date,
                                "confidence": confidence,
                                "comment": found_comment or "—"
                            })
                    else:
                        print(f"      Queued for consensus write.")
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
        from collections import Counter
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
                    "file": r['file'],
                    "raw_guess": r['raw_date'] or date,
                    "confidence": r['confidence'],
                    "comment": r['comment'] or "—"
                })

    write_review_report(folder, review_queue, root_folder=root_folder)

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
        from itertools import groupby
        global_offset = 0
        for subfolder, path_iter in groupby(all_paths, key=os.path.dirname):
            subfolder_files = [os.path.basename(p) for p in path_iter]
            print(f"\n📁 Processing folder: {subfolder} ({len(subfolder_files)} files)")
            _process_folder(subfolder, subfolder_files, cutoff_year, confidence_threshold, xmp_only, enable_geo,
                            global_offset=global_offset, global_total=global_total, folder_consensus=folder_consensus,
                            root_folder=folder, dry_run=dry_run, skip_dated=skip_dated)
            global_offset += len(subfolder_files)
        if not dry_run:
            _clear_checkpoint(folder)
        return

    files = natsorted([f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)])
    _process_folder(folder, files, cutoff_year, confidence_threshold, xmp_only, enable_geo, folder_consensus=folder_consensus, dry_run=dry_run, skip_dated=skip_dated)
    if not dry_run:
        _clear_checkpoint(folder)

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
            "  python metadata-ai.py /path/to/photos          # prompts for remaining options\n"
            "  python metadata-ai.py /path/to/photos -r --geotag --consensus\n"
            "  python metadata-ai.py /path/to/photos --cutoff 1995 --confidence 6 --xmp-only"
        )
    )
    parser.add_argument("directory", nargs="?", default=None,
                        help="Path to photos directory (prompted if omitted)")
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

    args = parser.parse_args()

    # Directory — prompt if not provided
    if args.directory:
        directory = re.sub(r'\\(.)', r'\1', args.directory.strip())
    else:
        raw = input(f"Enter photos directory [{DIRECTORY}]: ").strip()
        directory = re.sub(r'\\(.)', r'\1', raw) or DIRECTORY

    # Cutoff year
    if args.cutoff is not None:
        cutoff_year = args.cutoff
    else:
        cutoff_input = input("Skip photos dated from which year or later? [2010]: ").strip()
        try:
            cutoff_year = int(cutoff_input) if cutoff_input else 2010
        except ValueError:
            print("Invalid year, defaulting to 2010.")
            cutoff_year = 2010

    # Confidence threshold
    if args.confidence is not None:
        confidence_threshold = args.confidence
    else:
        conf_input = input("Confidence threshold for auto-write (1-10) [7]: ").strip()
        try:
            confidence_threshold = int(conf_input) if conf_input else 7
        except ValueError:
            confidence_threshold = 7

    # Boolean flags — skip prompts if provided via CLI
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

    if args.model:
        MODEL_ID = args.model
        print(f"Using model: {MODEL_ID}")
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE — no files will be modified.\n")

    # Check for an existing progress file and offer to resume
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
