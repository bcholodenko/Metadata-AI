import os
import io
import sys
import base64
import re
from datetime import datetime
from natsort import natsorted
from openai import OpenAI
import piexif
import piexif.helper
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

# Configuration
DIRECTORY = "./photos"         # Folder containing your images
MODEL_ID = "qwen/qwen3.6-27b" # Must match the model identifier in LM Studio
CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

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
    # First try exact YYYY:MM:DD
    m = re.search(r'(\d{4})[:/-](\d{2})[:/-](\d{2})', text)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)} 12:00:00", None

    # Then exact YYYY:MM
    m = re.search(r'(\d{4})[:/-](\d{2})', text)
    if m:
        return f"{m.group(1)}:{m.group(2)}:01 12:00:00", None

    # Then fuzzy
    for pattern, formatter in FUZZY_DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return formatter(m), text.strip()

    return None, None

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def get_jpeg_base64(image_path):
    """Converts any supported image to JPEG in memory. LM Studio only supports JPEG, PNG, WebP."""
    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def ask_vlm(image_path, prompt):
    """Sends an image to LM Studio as JPEG regardless of source format."""
    try:
        base64_image = get_jpeg_base64(image_path)
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
        print(f"   API Error: {e}")
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
    elif ext in ('.tiff', '.tif', '.png', '.heic', '.webp'):
        _apply_metadata_tiff(path, date_str, tags, comment, raw_date)
    else:
        _apply_metadata_jpeg(path, date_str, tags, comment, raw_date, gps)

def _apply_metadata_jpeg(path, date_str, tags=None, comment=None, raw_date=None, gps=None):
    try:
        exif_dict = piexif.load(path)
        if 'Exif' not in exif_dict:
            exif_dict['Exif'] = {}
        if '0th' not in exif_dict:
            exif_dict['0th'] = {}

        exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = date_str.encode('utf-8')

        user_comment_parts = []
        if comment:
            user_comment_parts.append(f"Comment: {comment}")
        if raw_date:
            user_comment_parts.append(f"Raw date: {raw_date}")
        if tags:
            user_comment_parts.append(f"Tags: {tags}")
        if user_comment_parts:
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(
                " | ".join(user_comment_parts), encoding="unicode"
            )

        if gps and 'GPS' not in exif_dict:
            exif_dict['GPS'] = {}
        if gps:
            lat, lon = gps
            def to_dms(val):
                d = int(abs(val))
                m = int((abs(val) - d) * 60)
                s = round(((abs(val) - d) * 60 - m) * 60 * 100)
                return [(d, 1), (m, 1), (s, 100)]
            exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = b'N' if lat >= 0 else b'S'
            exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = to_dms(lat)
            exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = b'E' if lon >= 0 else b'W'
            exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = to_dms(lon)

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, path)
        print(f"   💾 Success: {os.path.basename(path)} updated.")
    except Exception as e:
        print(f"   💾 Metadata Error for {os.path.basename(path)}: {e}")

def _apply_metadata_tiff(path, date_str, tags=None, comment=None, raw_date=None):
    try:
        img = Image.open(path)
        tiff_tags = img.tag_v2 if hasattr(img, 'tag_v2') else {}
        tiff_tags[306] = date_str
        parts = []
        if tags:
            parts.append(f"Tags: {tags}")
        if comment:
            parts.append(f"Comment: {comment}")
        if raw_date:
            parts.append(f"Raw date: {raw_date}")
        if parts:
            tiff_tags[270] = " | ".join(parts)
        img.save(path, tiffinfo=tiff_tags)
        print(f"   💾 Success: {os.path.basename(path)} updated.")
    except Exception as e:
        print(f"   💾 Metadata Error for {os.path.basename(path)}: {e}")

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
  <title>Metadata AI — Review Queue</title>
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
  <h1>📋 Metadata AI — Manual Review Queue</h1>
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
        from geopy.exc import GeocoderTimedOut
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
def process_archive(folder, cutoff_year=2010, confidence_threshold=7, xmp_only=False, enable_geo=False):
    if not os.path.exists(folder):
        print(f"Directory {folder} not found.")
        return

    files = natsorted([f for f in os.listdir(folder) if f.lower().endswith(
        ('.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.dng', '.webp'))])
    processed_files = set()
    review_queue = []

    print(f"Starting archival of {len(files)} photos...")

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

            is_back_prompt = (
                "Look very carefully at this image. Is it the BACK (reverse side) of a physical printed photograph? "
                "The back would show: blank paper, handwriting, stamps, photo lab printing, or a plain surface. "
                "If you see any photographic image content AT ALL, answer No. "
                "Answer ONLY 'Yes' or 'No'."
            )
            is_back_resp = ask_vlm(next_path, is_back_prompt)
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

        # Step 2: Check existing EXIF tags
        if not found_date:
            print(f"   2) Checking existing EXIF tags...")
            try:
                exif_data = piexif.load(current_path)
                comment_bytes = exif_data['Exif'].get(piexif.ExifIFD.UserComment, b'')
                comment = piexif.helper.UserComment.load(comment_bytes) if comment_bytes else ""
                found_date, raw_date_text = parse_fuzzy_date(str(comment))
                if found_date:
                    print(f"      Date from EXIF: {found_date}")
                else:
                    print(f"      No date in EXIF.")
            except:
                print(f"      Could not read EXIF.")

        # Step 3: VLM visual date guess with confidence score
        if not found_date:
            print(f"   3) Asking VLM to guess date from image content...")
            guess_prompt = (
                "Analyze the fashion, hairstyles, technology, and setting in this photo. "
                "Estimate the date as specifically as possible — could be YYYY:MM, YYYY, a decade like '1970s', or 'circa 1965'. "
                f"The date must be before {cutoff_year}. "
                "Also provide a confidence score from 1-10 for your estimate. "
                "Reply in this exact format:\nDATE: <your estimate>\nCONFIDENCE: <score>"
            )
            resp = ask_vlm(current_path, guess_prompt)

            date_line = re.search(r'DATE:\s*(.+)', resp)
            conf_line = re.search(r'CONFIDENCE:\s*(\d+)', resp)

            raw_guess = date_line.group(1).strip() if date_line else resp.strip()
            confidence = int(conf_line.group(1)) if conf_line else 5

            found_date, raw_date_text = parse_fuzzy_date(raw_guess)
            if found_date:
                print(f"      VLM guessed date: {found_date} (confidence: {confidence}/10)" + (f" — fuzzy: {raw_date_text}" if raw_date_text else ""))
            else:
                print(f"      VLM could not determine a date.")

        # Step 3b: Geotagging
        if enable_geo:
            print(f"   3b) Checking for location clues...")
            geo_prompt = (
                "Look at this photo for specific location clues — identifiable landmarks, street signs, place names, flags, or very distinctive geography. "
                "If you can name a specific city, region, or landmark with confidence, return ONLY that place name, nothing else. "
                "If there are no clear location clues, return ONLY the word 'none'."
            )
            geo_resp = ask_vlm(current_path, geo_prompt).strip()
            # Filter out non-answers: 'none', long explanatory sentences, or anything without a real place name
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
                print(f"      No location identified.")

        # Step 4: Write metadata
        print(f"   4) 💾 Writing metadata...")
        if found_date:
            try:
                year = int(found_date[:4])
                if year < cutoff_year:
                    if confidence >= confidence_threshold:
                        tags_resp = ask_vlm(current_path, "Describe this photo in 5 keywords, comma separated.")
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

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input(f"Enter photos directory [{DIRECTORY}]: ").strip().replace(chr(92) + " ", " ") or DIRECTORY

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

    process_archive(directory, cutoff_year, confidence_threshold, xmp_only, enable_geo)
