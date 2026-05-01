import os
import base64
import re
from natsort import natsorted
from openai import OpenAI
import piexif
import piexif.helper
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

# Configuration
DIRECTORY = "./photos" # Folder containing your images
MODEL_ID = "qwen/qwen3.6-27b" # Must match the model identifier in LM Studio
CLIENT = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_jpeg_base64(image_path):
    """Converts any supported image to JPEG in memory and returns base64. LM Studio only supports JPEG, PNG, WebP."""
    import io
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

def apply_metadata(path, date_str, tags=None):
    """Applies Date Taken and Tags to EXIF. Uses piexif for JPEG, Pillow for TIFF."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tiff', '.tif', '.png', '.heic', '.webp'):
        _apply_metadata_tiff(path, date_str, tags)
    elif ext == '.dng':
        _apply_metadata_dng(path, date_str, tags)
        _apply_metadata_tiff(path, date_str, tags)
    else:
        _apply_metadata_jpeg(path, date_str, tags)

def _apply_metadata_jpeg(path, date_str, tags=None):
    """Writes EXIF metadata to a JPEG using piexif."""
    try:
        exif_dict = piexif.load(path)

        if 'Exif' not in exif_dict:
            exif_dict['Exif'] = {}

        exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = date_str.encode('utf-8')

        if tags:
            exif_dict['Exif'][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(tags, encoding="unicode")

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, path)
        print(f"   Success: {os.path.basename(path)} updated.")
    except Exception as e:
        print(f"   Metadata Error for {os.path.basename(path)}: {e}")

def _apply_metadata_tiff(path, date_str, tags=None):
    """Writes EXIF metadata to a TIFF using Pillow."""
    try:
        img = Image.open(path)
        tiff_tags = img.tag_v2 if hasattr(img, 'tag_v2') else {}
        tiff_tags[306] = date_str          # Tag 306 = DateTime
        if tags:
            tiff_tags[270] = tags          # Tag 270 = ImageDescription (keywords)
        img.save(path, tiffinfo=tiff_tags)
        print(f"   Success: {os.path.basename(path)} updated.")
    except Exception as e:
        print(f"   Metadata Error for {os.path.basename(path)}: {e}")

def _apply_metadata_dng(path, date_str, tags=None, comment=None):
    """Writes metadata for DNG files via XMP sidecar."""
    try:
        xmp_path = os.path.splitext(path)[0] + ".xmp"
        keywords_xml = ""
        if tags:
            keywords_xml = "".join(
                f"      <rdf:li>{kw.strip()}</rdf:li>\n" for kw in tags.split(",")
            )
        comment_xml = f"  <dc:description><rdf:Alt><rdf:li xml:lang='x-default'>{comment}</rdf:li></rdf:Alt></dc:description>\n" if comment else ""
        xmp_content = f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
    <rdf:Description xmlns:xmp='http://ns.adobe.com/xap/1.0/'
                     xmlns:dc='http://purl.org/dc/elements/1.1/'
                     xmlns:exif='http://ns.adobe.com/exif/1.0/'>
      <exif:DateTimeOriginal>{date_str}</exif:DateTimeOriginal>
{comment_xml}      <dc:subject>
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
        print(f"   Metadata Error for {os.path.basename(path)}: {e}")


def process_archive(folder, cutoff_year=2010):
    if not os.path.exists(folder):
        print(f"Directory {folder} not found.")
        return

    files = natsorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.dng', '.webp'))])
    processed_files = set()
    
    print(f"Starting archival of {len(files)} photos...")

    for i in range(len(files)):
        current_file = files[i]
        if current_file in processed_files:
            continue

        current_path = os.path.join(folder, current_file)
        found_date = None
        found_comment = None

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

                ocr_prompt = "Extract any date written on this photo back. Return ONLY in YYYY:MM:DD format. If no date is present, return 'none'."
                resp = ask_vlm(next_path, ocr_prompt)
                date_match = re.search(r'(\d{4})[:/-](\d{2})[:/-](\d{2})', resp)
                if date_match:
                    found_date = f"{date_match.group(1)}:{date_match.group(2)}:{date_match.group(3)} 12:00:00"
                    print(f"      Date from back: {found_date}")
                else:
                    print(f"      No date on back — falling through to VLM guess.")

                # OCR any handwritten comments from the back, translate to English if needed
                comment_prompt = (
                    "Extract any handwritten or printed text from this photo back, excluding any dates. "
                    "If the text is not in English, translate it to English. "
                    "Return only the final English text, or 'none' if there is no text."
                )
                comment_resp = ask_vlm(next_path, comment_prompt)
                if comment_resp.strip().lower() != "none" and comment_resp.strip():
                    found_comment = comment_resp.strip()
                    print(f"      Comment from back: {found_comment}")
                else:
                    found_comment = None
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
                date_match = re.search(r'(\d{4})[:/-](\d{2})', str(comment))
                if date_match:
                    found_date = f"{date_match.group(1)}:{date_match.group(2)}:01 12:00:00"
                    print(f"      Date from EXIF: {found_date}")
                else:
                    print(f"      No date in EXIF.")
            except:
                print(f"      Could not read EXIF.")

        # Step 3: LLM Visual Guessing
        if not found_date:
            print(f"   3) Asking VLM to guess date from image content...")
            guess_prompt = "Analyze fashion and technology in this photo. Estimate year and month. Return ONLY YYYY:MM. Must be before 2010."
            resp = ask_vlm(current_path, guess_prompt)
            date_match = re.search(r'(\d{4})[:/-](\d{2})', resp)
            if date_match:
                found_date = f"{date_match.group(1)}:{date_match.group(2)}:01 12:00:00"
                print(f"      VLM guessed date: {found_date}")
            else:
                print(f"      VLM could not determine a date.")

        # Step 4: Write metadata
        print(f"   4) 💾 Writing metadata...")
        if found_date:
            try:
                year = int(found_date[:4])
                if year < cutoff_year:
                    tags_resp = ask_vlm(current_path, "Describe this photo in 5 keywords, comma separated.")
                    combined_tags = f"{tags_resp} | {found_comment}" if found_comment else tags_resp
                    apply_metadata(current_path, found_date, combined_tags)
                else:
                    print(f"      ⏭️  Skipping: date {year} is {cutoff_year} or later.")
            except: pass
        else:
            print(f"      ❌ No date found — skipping.")

if __name__ == "__main__":
    import sys
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

    process_archive(directory, cutoff_year)
