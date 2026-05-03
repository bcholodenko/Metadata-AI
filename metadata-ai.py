import os
import io
import sys
import base64
import re

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

EXTENSIONS = ('.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.dng', '.webp')

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
    m = re.search(r'(\d{4})[:/-](\d{2})[:/-](\d{2})', text)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)} 12:00:00", None
    m = re.search(r'(\d{4})[:/-](\d{2})', text)
    if m:
        return f"{m.group(1)}:{m.group(2)}:01 12:00:00", None
    # M-D-YY or M-D-YYYY folder name patterns e.g. "7-3-87" or "8-26 to 8-30-87 Hawaii"
    m = re.search(r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{2})\b', text)
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        full_year = f"19{year}" if int(year) > 20 else f"20{year}"
        return f"{full_year}:{int(month):02d}:{int(day):02d} 12:00:00", None
    m = re.search(r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b', text)
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        return f"{year}:{int(month):02d}:{int(day):02d} 12:00:00", None
    # Bare 4-digit year
    m = re.search(r'\b(\d{4})\b', text)
    if m:
        return f"{m.group(1)}:01:01 12:00:00", None
    for pattern, formatter in FUZZY_DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return formatter(m), text.strip()
    return None, None

# ---------------------------------------------------------------------------
# IPTC helper
# ---------------------------------------------------------------------------
def get_iptc_metadata(path):
    """Extracts existing IPTC metadata for checking and VLM context."""
    try:
        info = IPTCInfo(path, force=True)
        raw_date = info['date created'].decode('utf-8') if info['date created'] else None
        caption = info['caption/abstract'].decode('utf-8') if info['caption/abstract'] else None
        keywords = [k.decode('utf-8') for k in info['keywords']] if info['keywords'] else []

        formatted_date = None
        if raw_date and len(raw_date) == 8:
            formatted_date = f"{raw_date[:4]}:{raw_date[4:6]}:{raw_date[6:8]} 12:00:00"

        return formatted_date, caption, keywords
    except Exception:
        return None, None, []

# ---------------------------------------------------------------------------
# Image & API helpers
# ---------------------------------------------------------------------------
def get_jpeg_base64(image_path):
    """
    Opens an image, downscales it so the long edge is at most VLM_MAX_DIMENSION,
    converts to JPEG, and returns a base64 string.

    Large scanned photos (600+ DPI) can easily exceed 100 MP. Pillow's decompression
    bomb guard is disabled at module level (Image.MAX_IMAGE_PIXELS = None), so we
    explicitly cap the resolution here before encoding — keeping memory use low and
    avoiding unnecessary data being sent to the VLM.
    """
    import warnings
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
        img = Image.open(image_path)
        return pytesseract.image_to_string(img).strip()
    except ImportError:
        return None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Metadata writers
# ---------------------------------------------------------------------------
def apply_metadata(path, date_str, tags=None, comment=None, raw_date=None, gps=None, xmp_only=False):
    ext = os.path.splitext(path)[1].lower()
    if xmp_only or ext == '.dng':
        _apply_metadata_xmp(path, date_str, tags, comment, raw_date, gps)
    elif ext in ('.jpg', '.jpeg'):
        _apply_metadata_jpeg(path, date_str, tags, comment, raw_date, gps)
    elif ext in ('.tiff', '.tif'):
        _apply_metadata_tiff(path, date_str, tags, comment, raw_date, gps)
    elif ext in ('.png', '.heic', '.webp'):
        _apply_metadata_png(path, date_str, tags, comment, raw_date, gps)
    else:
        print(f"   ⚠️ Unsupported format for metadata writing: {ext}")

def _apply_metadata_jpeg(path, date_str, tags=None, comment=None, raw_date=None, gps=None):
    # XMP+ExifTool approach for consistency and to avoid any
    import shutil, subprocess
    xmp_path = os.path.splitext(path)[0] + ".xmp"
    _apply_metadata_xmp(path, date_str, tags, comment, raw_date, gps)

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
            print(f"   💾 Success: {os.path.basename(path)} updated via ExifTool.")
        else:
            print(f"      ⚠️ ExifTool merge failed — XMP sidecar kept. Error: {result.stderr.strip()}")
    except Exception as e:
        print(f"      ⚠️ ExifTool error — XMP sidecar kept: {e}")

def _apply_metadata_tiff(path, date_str, tags=None, comment=None, raw_date=None, gps=None):
    # Write XMP sidecar first, then use ExifTool to merge it safely into the TIFF's
    # EXIF without re-encoding any pixel data. If ExifTool is not available, the XMP
    # sidecar is kept as a fallback.
    import shutil, subprocess
    xmp_path = os.path.splitext(path)[0] + ".xmp"
    _apply_metadata_xmp(path, date_str, tags, comment, raw_date, gps)

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
            print(f"   💾 Success: {os.path.basename(path)} updated via ExifTool.")
        else:
            print(f"      ⚠️ ExifTool merge failed — XMP sidecar kept. Error: {result.stderr.strip()}")
    except Exception as e:
        print(f"      ⚠️ ExifTool error — XMP sidecar kept: {e}")


def _apply_metadata_png(path, date_str, tags=None, comment=None, raw_date=None, gps=None):
    # Same XMP+ExifTool approach as TIFF — avoids re-encoding pixel data.
    import shutil, subprocess
    xmp_path = os.path.splitext(path)[0] + ".xmp"
    _apply_metadata_xmp(path, date_str, tags, comment, raw_date, gps)

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
            print(f"   💾 Success: {os.path.basename(path)} updated via ExifTool.")
        else:
            print(f"      ⚠️ ExifTool merge failed — XMP sidecar kept. Error: {result.stderr.strip()}")
    except Exception as e:
        print(f"      ⚠️ ExifTool error — XMP sidecar kept: {e}")

def _apply_metadata_xmp(path, date_str, tags=None, comment=None, raw_date=None, gps=None):
    try:
        xmp_path = os.path.splitext(path)[0] + ".xmp"
        keywords_xml = ""
        if tags:
            keywords_xml = "".join(
                f"          <rdf:li>{kw.strip()}</rdf:li>\n" for kw in tags.split(",")
            )
        comment_parts = []
        if comment:
            comment_parts.append(f"Comment: {comment}")
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
        print(f"   💾 Success: {os.path.basename(xmp_path)} written.")
    except Exception as e:
        print(f"   💾 Metadata Error for {os.path.basename(path)}: {e}")

# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------
def write_review_report(folder, review_queue):
    """Writes an HTML report of low-confidence photos for manual review."""
    if not review_queue:
        return
    report_path = os.path.join(folder, "review.html")
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

def _process_folder(folder, files, cutoff_year, confidence_threshold, xmp_only, enable_geo):
    processed_files = set()
    review_queue = []

    # Extract date and location hint from folder name
    folder_name = os.path.basename(folder)
    folder_date, folder_date_raw = parse_fuzzy_date(folder_name)
    # Simple location hint: words after stripping date-like tokens
    folder_location = re.sub(r'[\d]{1,4}[-/ to]+[\d]{1,4}(?:[-/ to]+[\d]{1,4})?', '', folder_name).strip(' -_')
    folder_location = folder_location if len(folder_location) > 2 else None
    if folder_date:
        print(f"   📁 Folder date hint: {folder_date}" + (f" — location hint: {folder_location}" if folder_location else ""))

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

        print(f"\n[{i+1}/{len(files)}] Processing: {current_file}")

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

            is_back_prompt = (
                "Look very carefully at this image. Is it the BACK (reverse side) of a physical printed photograph? "
                "The back would show: blank paper, handwriting, stamps, photo lab printing, or a plain surface. "
                "If you see any photographic image content AT ALL, answer No. "
                "Answer ONLY 'Yes' or 'No'."
            )
            is_back_resp = ask_vlm(next_path, is_back_prompt) if next_readable else ""  
            is_back = False
            if is_back_resp.strip().lower().startswith("yes"):
                confirm_prompt = "Does this image show a photographic scene with people, places, or objects? Answer ONLY 'Yes' or 'No'."
                confirm_resp = ask_vlm(next_path, confirm_prompt)
                if not confirm_resp.strip().lower().startswith("yes"):
                    is_back = True

            if is_back:
                back_confirmed = True
                print(f"      Back confirmed: {next_file}")
                processed_files.add(next_file)

                # Run Tesseract first if available, then feed to VLM
                ocr_context = run_tesseract(next_path)
                context_str = f"\n\nOCR pre-extracted text from this image:\n{ocr_context}" if ocr_context else ""

                ocr_prompt = (
                    f"Extract any date written on this photo back.{context_str} "
                    "Return ONLY in YYYY:MM:DD format, or a description like 'circa 1950s', 'early 1970s'. "
                    "If no date is present, return 'none'."
                )
                resp = ask_vlm(next_path, ocr_prompt)
                found_date, raw_date_text = parse_fuzzy_date(resp)
                if found_date:
                    print(f"      Date from back: {found_date}" + (f" (fuzzy: {raw_date_text})" if raw_date_text else ""))
                else:
                    print(f"      No date on back — falling through to VLM guess.")

                # Extract comments
                comment_prompt = (
                    f"Extract any handwritten or printed text from this photo back, excluding any dates.{context_str} "
                    "If the text is not in English, translate it to English. "
                    "Return only the final English text, or 'none' if there is no text."
                )
                comment_resp = ask_vlm(next_path, comment_prompt)
                if comment_resp.strip().lower() != "none" and comment_resp.strip():
                    found_comment = comment_resp.strip()
                    print(f"      Comment from back: {found_comment}")
                else:
                    print(f"      No comment on back.")
            else:
                print(f"      No back detected.")
        else:
            print(f"      No next image to check.")

        # Step 2: Check folder name then IPTC keywords for a date
        if not found_date:
            print(f"   2) Checking folder name and IPTC keywords for date...")
            if folder_date:
                found_date = folder_date
                raw_date_text = folder_date_raw
                confidence = 10
                print(f"      Date from folder name '{folder_name}': {found_date}")
            else:
                _, _, iptc_keywords = get_iptc_metadata(current_path)
                if iptc_keywords:
                    for keyword in iptc_keywords:
                        found_date, raw_date_text = parse_fuzzy_date(keyword)
                        if found_date:
                            print(f"      Date parsed from IPTC keyword '{keyword}': {found_date}" + (f" (fuzzy: {raw_date_text})" if raw_date_text else ""))
                            break
                    if not found_date:
                        print(f"      No date found in IPTC keywords.")
                else:
                    print(f"      No IPTC keywords found.")

        # Step 3: VLM visual date guess (and always time-of-day estimation)
        # If date already known, skip date guessing but still ask for time of day.
        raw_time = None
        if not found_date:
            print(f"   3) Asking VLM to guess date from image content...")
            folder_context = ""
            if folder_date:
                folder_context += f"The folder containing this photo is named '{folder_name}', suggesting the date is around {folder_date[:4]}. "
            if folder_location:
                folder_context += f"The folder name also suggests the location may be '{folder_location}'. "
            guess_prompt = (
                "Analyze the fashion, hairstyles, technology, and setting in this photo. "
                f"{folder_context}"
                "Estimate the date as specifically as possible — could be YYYY:MM, YYYY, a decade like '1970s', or 'circa 1965'. "
                f"The date must be before {cutoff_year}. "
                "Also estimate the time of day based on lighting, shadows, and context — e.g. 'morning', 'midday', 'afternoon', 'evening', or a specific time like '3pm'. "
                "Also provide a confidence score from 1-10 for your estimate. "
                "Reply in this exact format:\nDATE: <your estimate>\nTIME: <your estimate>\nCONFIDENCE: <score>"
            )
            resp = ask_vlm(current_path, guess_prompt)

            date_line = re.search(r'DATE:\s*(.+)', resp)
            time_line = re.search(r'TIME:\s*(.+)', resp)
            conf_line = re.search(r'CONFIDENCE:\s*(\d+)', resp)

            raw_guess = date_line.group(1).strip() if date_line else resp.strip()
            raw_time = time_line.group(1).strip() if time_line else None
            confidence = int(conf_line.group(1)) if conf_line else 5

            found_date, raw_date_text = parse_fuzzy_date(raw_guess)

            if found_date:
                time_str = f" (~{raw_time})" if raw_time else ""
                print(f"      VLM guessed date: {found_date} (confidence: {confidence}/10){time_str}" + (f" — fuzzy: {raw_date_text}" if raw_date_text else ""))
            else:
                print(f"      VLM could not determine a date. Raw response: '{raw_guess}' (confidence: {confidence}/10)")
        else:
            print(f"   3) Asking VLM to estimate time of day...")
            time_prompt = (
                "Look at the lighting, shadows, and context in this photo. "
                "Estimate the time of day — e.g. 'morning', 'midday', 'afternoon', 'evening', or a specific time like '3pm'. "
                "Reply in this exact format:\nTIME: <your estimate>"
            )
            time_resp = ask_vlm(current_path, time_prompt)
            time_line = re.search(r'TIME:\s*(.+)', time_resp)
            raw_time = time_line.group(1).strip() if time_line else None
            if raw_time:
                print(f"      VLM estimated time: {raw_time}")
            else:
                print(f"      VLM could not estimate time of day.")

        # Apply estimated time of day to the date string
        time_hour = _parse_time_of_day(raw_time) if raw_time else 12
        if found_date:
            found_date = found_date[:11] + f"{time_hour:02d}:00:00"
            if raw_time:
                time_str = f" (~{raw_time})"
                print(f"      Final timestamp: {found_date}{time_str}")

        # Step 4: Geotagging
        if enable_geo:
            print(f"   4) Checking for location clues...")
            geo_prompt = (
                "Look at this photo for specific location clues — identifiable landmarks, street signs, place names, flags, or very distinctive geography. "
                "If you can name a specific city, region, or landmark with confidence, return ONLY that place name, nothing else. "
                "If there are no clear location clues, return ONLY the word 'none'."
            )
            geo_resp = ask_vlm(current_path, geo_prompt).strip()
            is_valid_location = (
                geo_resp.lower() != "none"
                and len(geo_resp) < 100
                and not any(phrase in geo_resp.lower() for phrase in [
                    "no identifiable", "no clear", "cannot identify", "unable to",
                    "no location", "no specific", "there are no", "i cannot", "i can't"
                ])
            )
            if is_valid_location:
                print(f"      Location identified: {geo_resp}")
                gps_coords = geolocate(geo_resp)
                if gps_coords:
                    print(f"      GPS: {gps_coords[0]:.4f}, {gps_coords[1]:.4f}")
                else:
                    print(f"      Could not resolve GPS — storing as text tag.")
                    found_comment = (found_comment + f" | Location: {geo_resp}") if found_comment else f"Location: {geo_resp}"
            else:
                if folder_location and not gps_coords:
                    print(f"      No location from image — trying folder name hint: '{folder_location}'")
                    gps_coords = geolocate(folder_location)
                    if gps_coords:
                        print(f"      GPS from folder name: {gps_coords[0]:.4f}, {gps_coords[1]:.4f}")
                        found_comment = (found_comment + f" | Location: {folder_location}") if found_comment else f"Location: {folder_location}"
                    else:
                        print(f"      No location identified.")
                else:
                    print(f"      No location identified.")

        # Step 5: Generate keywords
        tags_resp = None
        if found_date:
            try:
                year = int(found_date[:4])
                if year < cutoff_year and confidence >= confidence_threshold:
                    print(f"   5) Generating keywords...")
                    tags_resp = ask_vlm(current_path, "Describe this photo in 5 keywords, comma separated.")
                    if tags_resp:
                        print(f"      Keywords: {tags_resp.strip()}")
            except:
                pass

        # Step 6: Write metadata
        print(f"   6) 💾 Writing metadata...")
        if found_date:
            try:
                year = int(found_date[:4])
                if year < cutoff_year:
                    if confidence >= confidence_threshold:
                        apply_metadata(current_path, found_date, tags=tags_resp, comment=found_comment,
                                       raw_date=raw_date_text, gps=gps_coords, xmp_only=xmp_only)
                    else:
                        print(f"      ⚠️  Low confidence ({confidence}/10) — added to review queue.")
                        review_queue.append({
                            "file": current_file,
                            "raw_guess": raw_date_text or found_date,
                            "confidence": confidence,
                            "comment": found_comment or "—"
                        })
                else:
                    print(f"      ⏭️  Skipping: date {year} is {cutoff_year} or later.")
            except:
                pass
        else:
            print(f"      ❌ No date found — skipping.")

    write_review_report(folder, review_queue)


def process_archive(folder, cutoff_year=2010, confidence_threshold=7, xmp_only=False, enable_geo=False, recursive=False):
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
        from itertools import groupby
        for subfolder, path_iter in groupby(all_paths, key=os.path.dirname):
            subfolder_files = [os.path.basename(p) for p in path_iter]
            print(f"\n📁 Processing folder: {subfolder} ({len(subfolder_files)} files)")
            _process_folder(subfolder, subfolder_files, cutoff_year, confidence_threshold, xmp_only, enable_geo)
        return

    files = natsorted([f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)])
    _process_folder(folder, files, cutoff_year, confidence_threshold, xmp_only, enable_geo)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        raw = input(f"Enter photos directory [{DIRECTORY}]: ").strip()
        import re as _re
        directory = _re.sub(r'\\(.)', r'\1', raw) or DIRECTORY

    cutoff_input = input("Skip photos dated from which year or later? [2010]: ").strip()
    try:
        cutoff_year = int(cutoff_input) if cutoff_input else 2010
    except ValueError:
        print("Invalid year, defaulting to 2010.")
        cutoff_year = 2010

    conf_input = input("Confidence threshold for auto-write (1-10) [7]: ").strip()
    try:
        confidence_threshold = int(conf_input) if conf_input else 7
    except ValueError:
        confidence_threshold = 7

    xmp_only = input("Write metadata to XMP sidecar files only? [y/N]: ").strip().lower() == "y"
    enable_geo = input("Enable geotagging? [y/N]: ").strip().lower() == "y"
    recursive = input("Recursively process subfolders? [y/N]: ").strip().lower() == "y"

    process_archive(directory, cutoff_year, confidence_threshold, xmp_only, enable_geo, recursive)
