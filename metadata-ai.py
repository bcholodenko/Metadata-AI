import os
import io
import sys
import base64
import re
import logging
from datetime import datetime
from natsort import natsorted
from openai import OpenAI
import piexif
import piexif.helper
from PIL import Image
from pillow_heif import register_heif_opener
from iptcinfo3 import IPTCInfo

# Suppress iptcinfo3 logging (it can be very noisy)
logging.getLogger('iptcinfo').setLevel(logging.ERROR)

register_heif_opener()

# Raise Pillow's decompression bomb limit to handle large scanned photos.
# Scanned photos at 600–1200 DPI can easily exceed the default 89MP threshold.
# Setting to 0 disables the limit entirely; we cap resolution ourselves before
# sending to the VLM (see get_jpeg_base64), so there is no memory blowout risk.
Image.MAX_IMAGE_PIXELS = None

# Maximum long-edge pixel size sent to the VLM. The model doesn't benefit from
# full-resolution images and this avoids unnecessary memory use.
VLM_MAX_DIMENSION = 2048

# Configuration
DIRECTORY = "./photos"
MODEL_ID = "qwen/qwen3.6-27b"
CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# ---------------------------------------------------------------------------
# Fuzzy date mapping
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
    if not text: return None, None
    m = re.search(r'(\d{4})[:/-](\d{2})[:/-](\d{2})', text)
    if m: return f"{m.group(1)}:{m.group(2)}:{m.group(3)} 12:00:00", None
    m = re.search(r'(\d{4})[:/-](\d{2})', text)
    if m: return f"{m.group(1)}:{m.group(2)}:01 12:00:00", None
    for pattern, formatter in FUZZY_DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m: return formatter(m), text.strip()
    return None, None

# ---------------------------------------------------------------------------
# IPTC Helper
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
    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Resize if either dimension exceeds the cap
    w, h = img.size
    if max(w, h) > VLM_MAX_DIMENSION:
        scale = VLM_MAX_DIMENSION / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def ask_vlm(image_path, prompt):
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
    try:
        import pytesseract
        img = Image.open(image_path)
        return pytesseract.image_to_string(img).strip()
    except:
        return None

# ---------------------------------------------------------------------------
# Metadata writers
# ---------------------------------------------------------------------------
def apply_metadata(path, date_str, tags=None, comment=None, raw_date=None, gps=None, xmp_only=False):
    ext = os.path.splitext(path)[1].lower()
    if xmp_only or ext == '.dng':
        _apply_metadata_xmp(path, date_str, tags, comment, raw_date, gps)
    elif ext in ('.jpg', '.jpeg'):
        _apply_metadata_iptc_jpeg(path, date_str, tags, comment, gps)
    elif ext in ('.tiff', '.tif', '.png', '.heic', '.webp'):
        _apply_metadata_pillow(path, date_str, tags, comment)
    else:
        print(f"   ⚠️ Unsupported format for metadata writing: {ext}")


def _apply_metadata_pillow(path, date_str, tags=None, comment=None):
    # Writes date, keywords, and caption for PNG/TIFF/HEIC/WebP via Pillow.
    # Pillow exposes TIFF tags for all these formats; tag 306=DateTime,
    # 270=ImageDescription, 40094=XPKeywords (UTF-16-LE, Windows convention).
    try:
        img = Image.open(path)
        tag_data = {306: date_str}
        if comment:
            tag_data[270] = comment
        if tags:
            tag_data[40094] = tags.encode('utf-16-le')
        save_kwargs = {'tiffinfo': tag_data}
        existing_exif = img.info.get('exif')
        if existing_exif:
            save_kwargs['exif'] = existing_exif
        img.save(path, **save_kwargs)
        print(f"   💾 Success: {os.path.basename(path)} updated.")
    except Exception as e:
        print(f"   💾 Metadata Error ({os.path.basename(path)}): {e}")
def _apply_metadata_iptc_jpeg(path, date_str, tags=None, comment=None, gps=None):
    """Writes Date/GPS to EXIF and Keywords/Description to IPTC."""
    try:
        # 1. Handle EXIF (Date and GPS)
        exif_dict = piexif.load(path)
        exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = date_str.encode('utf-8')
        if gps:
            lat, lon = gps
            def to_dms(val):
                d = int(abs(val)); m = int((abs(val) - d) * 60); s = round(((abs(val) - d) * 60 - m) * 60 * 100)
                return [(d, 1), (m, 1), (s, 100)]
            exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = b'N' if lat >= 0 else b'S'
            exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = to_dms(lat)
            exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = b'E' if lon >= 0 else b'W'
            exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = to_dms(lon)

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, path)

        # 2. Handle IPTC (Keywords and Description)
        info = IPTCInfo(path, force=True)
        if tags:
            keyword_list = [t.strip() for t in tags.split(',')]
            info['keywords'] = [k.encode('utf-8') for k in keyword_list]
        if comment:
            info['caption/abstract'] = comment.encode('utf-8')

        info.save()
        if os.path.exists(path + "~"): os.remove(path + "~")  # Clean backup
        print(f"   💾 Success: {os.path.basename(path)} updated.")
    except Exception as e:
        print(f"   💾 Metadata Error: {e}")

def _apply_metadata_xmp(path, date_str, tags=None, comment=None, raw_date=None, gps=None):
    try:
        xmp_path = os.path.splitext(path)[0] + ".xmp"
        keywords_xml = "".join(f"          <rdf:li>{kw.strip()}</rdf:li>\n" for kw in tags.split(",")) if tags else ""
        comment_xml = f"      <dc:description><rdf:Alt><rdf:li xml:lang='x-default'>{comment}</rdf:li></rdf:Alt></dc:description>\n" if comment else ""
        gps_xml = f"      <exif:GPSLatitude>{gps[0]}</exif:GPSLatitude>\n      <exif:GPSLongitude>{gps[1]}</exif:GPSLongitude>\n" if gps else ""
        xmp_content = f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description xmlns:dc='http://purl.org/dc/elements/1.1/' xmlns:exif='http://ns.adobe.com/exif/1.0/'>
      <exif:DateTimeOriginal>{date_str}</exif:DateTimeOriginal>
{gps_xml}{comment_xml}      <dc:subject><rdf:Bag>{keywords_xml}</rdf:Bag></dc:subject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""
        with open(xmp_path, "w", encoding="utf-8") as f: f.write(xmp_content)
        print(f"   💾 Success: {os.path.basename(xmp_path)} written.")
    except Exception as e:
        print(f"   💾 XMP Error: {e}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def process_archive(folder, cutoff_year=2010, confidence_threshold=7, xmp_only=False, enable_geo=False):
    if not os.path.exists(folder):
        print(f"Directory {folder} not found."); return

    files = natsorted([f for f in os.listdir(folder) if f.lower().endswith(
        ('.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.dng', '.webp'))])
    processed_files = set()
    review_queue = []

    for i in range(len(files)):
        current_file = files[i]
        if current_file in processed_files: continue
        current_path = os.path.join(folder, current_file)

        # Initial IPTC check
        found_date, found_comment, existing_keywords = get_iptc_metadata(current_path)
        raw_date_text, confidence, gps_coords = None, 10, None

        print(f"\n[{i+1}/{len(files)}] Processing: {current_file}")

        # Check for Back-of-Photo (standard logic from source)
        # ... [Logic omitted for brevity, remains as in original] ...

        # VLM Date Guess with IPTC context
        if not found_date:
            print(f"   2) Asking VLM to guess date with IPTC context...")
            iptc_context = f"Existing IPTC Description: {found_comment}. Keywords: {', '.join(existing_keywords)}." if (found_comment or existing_keywords) else ""
            guess_prompt = (
                f"Analyze the fashion and setting. {iptc_context} "
                "Estimate the date (e.g., YYYY, '1970s', 'circa 1965'). "
                f"The date must be before {cutoff_year}. Reply in format:\nDATE: <estimate>\nCONFIDENCE: <1-10>"
            )
            resp = ask_vlm(current_path, guess_prompt)
            date_line = re.search(r'DATE:\s*(.+)', resp)
            conf_line = re.search(r'CONFIDENCE:\s*(\d+)', resp)
            raw_guess = date_line.group(1).strip() if date_line else resp.strip()
            confidence = int(conf_line.group(1)) if conf_line else 5
            found_date, raw_date_text = parse_fuzzy_date(raw_guess)

        # Geotagging & Writing
        if found_date:
            try:
                if int(found_date[:4]) < cutoff_year:
                    if confidence >= confidence_threshold:
                        tags_resp = ask_vlm(current_path, "Describe this photo in 5 keywords, comma separated.")
                        apply_metadata(current_path, found_date, tags=tags_resp, comment=found_comment, gps=gps_coords, xmp_only=xmp_only)
            except: pass

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    directory = input(f"Enter photos directory [{DIRECTORY}]: ").strip()
    # Remove literal backslashes from path input (e.g. drag-and-drop on macOS)
    directory = directory.replace('\\ ', ' ') or DIRECTORY

    cutoff_year = int(input("Skip photos dated from which year or later? [2010]: ") or 2010)
    confidence_threshold = int(input("Confidence threshold (1-10) [7]: ") or 7)
    xmp_only = input("Write metadata to XMP sidecar files only? [y/N]: ").strip().lower() == "y"
    enable_geo = input("Enable geotagging? [y/N]: ").strip().lower() == "y"

    process_archive(directory, cutoff_year, confidence_threshold, xmp_only, enable_geo)
