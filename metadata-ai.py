import os
import io
import sys
import base64
import re
import shlex
import time
import warnings
import json
import shutil
import subprocess
import tempfile
import textwrap
from itertools import groupby
from pathlib import Path
from collections import Counter, deque

# Force line-buffered output so progress prints appear immediately in the terminal
sys.stdout.reconfigure(line_buffering=True)
import logging
from datetime import datetime
from natsort import natsorted
from openai import OpenAI
from PIL import Image
from pillow_heif import register_heif_opener
from iptcinfo3 import IPTCInfo
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Suppress iptcinfo3 logging (it can be very noisy)
logging.getLogger('iptcinfo').setLevel(logging.ERROR)

register_heif_opener()

console = Console()

def _yn(prompt: str, default_yes: bool = False) -> bool:
    """Styled y/n prompt. Falls back cleanly if stdin is not a tty."""
    hint = "[Y/n]" if default_yes else "[y/N]"
    try:
        ans = console.input(f"\n[bold]{prompt}[/bold] {hint} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default_yes
    if not ans:
        return default_yes
    return ans.startswith("y")

def _ask(prompt: str, default: str = "") -> str:
    """Styled text prompt with a default value shown in dim."""
    hint = f" [[dim]{default}[/dim]]" if default else ""
    try:
        val = console.input(f"[bold]{prompt}[/bold]{hint} ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return val or default

def _questionary_select(prompt: str, choices: list[tuple[str, str]], default: str) -> str:
    """Use questionary for a styled select if available; fall back to numbered list."""
    try:
        import questionary
        if sys.stdin.isatty():
            q_choices = [questionary.Choice(title=f"{k:<22} {desc}", value=k) for k, desc in choices]
            result = questionary.select(prompt, choices=q_choices, default=default).ask()
            return result if result else default
    except ImportError:
        pass
    # Fallback: numbered list
    console.print(f"\n[bold]{prompt}[/bold]")
    for i, (k, desc) in enumerate(choices, 1):
        marker = "[cyan]›[/cyan]" if k == default else " "
        console.print(f"  {marker} [bold]{i}.[/bold] [cyan]{k:<22}[/cyan] {desc}")
    try:
        raw = console.input(f"Choice [1–{len(choices)}] [[dim]{default}[/dim]]: ").strip()
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx][0]
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    return default

def _yn_select(prompt: str, default_yes: bool = False) -> bool:
    """Yes/No prompt rendered as an arrow-key questionary select.

    Falls back to _yn (styled text prompt) when questionary is not installed
    or stdin is not a tty (e.g. piped input or CI environments).
    """
    try:
        import questionary
        if sys.stdin.isatty():
            yes_choice = questionary.Choice("Yes", value=True)
            no_choice  = questionary.Choice("No",  value=False)
            result = questionary.select(
                prompt,
                choices=[yes_choice, no_choice],
                default=yes_choice if default_yes else no_choice,
            ).ask()
            return result if result is not None else default_yes
    except ImportError:
        pass
    return _yn(prompt, default_yes=default_yes)

def _format_eta(seconds: float) -> str:
    """Format a seconds value as a human-readable ETA string.

    Scales from seconds up through minutes, hours, and days so the display
    is always readable regardless of run length.
    """
    if seconds < 60:
        return f"~{int(seconds)}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"~{m}m {s:02d}s"
    if seconds < 86400:
        h, rem = divmod(int(seconds), 3600)
        m = rem // 60
        return f"~{h}h {m:02d}m"
    d, rem = divmod(int(seconds), 86400)
    h = rem // 3600
    return f"~{d}d {h:02d}h"

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
        console.print(f"      [red]Could not open image {os.path.basename(image_path)}: {e}[/red]")
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
        console.print(f"      [red]VLM request failed: {e}[/red]")
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

    Recognizes:
      - 4-digit years anywhere ('Christmas 1978', 'Summer Vacation 1965')
      - Short date formats common in scan organization:
          M-D-YY, MM-DD-YY    e.g. '6-14-82', '08-15-92'
          M/D/YY, MM/DD/YY    e.g. '2/12/87'
          M-YY, MM-YY         e.g. '8-79', '12-99'  (M/YY also accepted)
      - Month name + 2-digit year e.g. 'Aug 87', 'January 92'
      - Folders with at least 2 alphabetic words ('Paris Trip')
    Rejects names matching the noise list ('New Folder (2)', 'Scans_Batch_3', etc.).
    """
    if not name or len(name) < 2:
        return False
    lower = name.lower().strip()
    for pattern in _FOLDER_NOISE_PATTERNS:
        if re.search(pattern, lower):
            return False

    # 4-digit year present anywhere
    if re.search(r'\b(18|19|20)\d{2}\b', name):
        return True

    # Numeric short-date formats with separators (M-D-YY, M/YY, etc.)
    # Accepts 1-2 digit month, optional 1-2 digit day, 2 or 4-digit year.
    if re.search(r'\b\d{1,2}[-/]\d{1,2}([-/]\d{2,4})?\b', name):
        return True

    # Month name followed by a year (2 or 4-digit), e.g. "Aug 87" or "January 1992"
    if re.search(
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|'
        r'january|february|march|april|june|july|august|september|october|november|december)'
        r'[\s.,-]+\d{2,4}\b',
        lower,
    ):
        return True

    # Two or more alphabetic words of >=3 chars (e.g. 'Paris Trip')
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
        console.print(f"   [yellow]⚠ Unsupported format for metadata writing: {ext}[/yellow]")

def _apply_metadata_via_exiftool(path, date_str, tags=None, comment=None, raw_date=None, gps=None, scene=None, setting=None, flash=None):
    """Write XMP sidecar then merge it into the file via ExifTool.
    Used for JPEG, TIFF, PNG, HEIC, WebP — anything where we don't want to re-encode pixels.
    The XMP sidecar is deleted after a successful merge; kept as a fallback if ExifTool is missing or fails."""
    xmp_path = os.path.splitext(path)[0] + ".xmp"
    # verbose=False — the XMP is a temp file about to be merged; the user only
    # sees the final outcome message below (or the warning if it falls back).
    _apply_metadata_xmp(path, date_str, tags, comment, raw_date, gps, scene, setting, flash, verbose=False)

    if shutil.which("exiftool") is None:
        console.print(f"      [yellow]⚠ ExifTool not found — XMP sidecar kept at {os.path.basename(xmp_path)}[/yellow]")
        return

    try:
        result = subprocess.run(
            ["exiftool", "-overwrite_original", f"-tagsfromfile={xmp_path}", path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            os.remove(xmp_path)
            console.print(f"   [bold green]✓[/bold green] {os.path.basename(path)}")
        else:
            console.print(f"      [yellow]⚠ ExifTool merge failed — XMP sidecar kept. Error: {result.stderr.strip()}[/yellow]")
    except Exception as e:
        console.print(f"      [yellow]⚠ ExifTool error — XMP sidecar kept: {e}[/yellow]")

def _xml_escape(s):
    """Minimal XML escape for text content."""
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))

def _apply_metadata_xmp(path, date_str, tags=None, comment=None, raw_date=None, gps=None, scene=None, setting=None, flash=None, verbose=True):
    """Write the XMP sidecar for `path`.

    When `verbose=False` the success print is suppressed — used when this function
    is called as an intermediate step before ExifTool merges the sidecar into the
    image file (the user only cares about the final outcome, not the temp file).
    """
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
      {"<exif:DateTimeOriginal>" + date_str + "</exif:DateTimeOriginal>" if date_str else ""}
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
        if verbose:
            console.print(f"   [bold green]✓[/bold green] {os.path.basename(xmp_path)}")
    except Exception as e:
        console.print(f"   [red]✗ Metadata Error for {os.path.basename(path)}: {e}[/red]")

# ---------------------------------------------------------------------------
# Review queue
#
# Two artifacts are produced for low-confidence photos:
#   - review.json: the canonical record. Each entry carries every field needed
#                  to re-run a metadata write, plus a status (pending/applied/
#                  skipped) so the review pass can resume across Ctrl-C.
#   - review.html: a visual reference with thumbnails. Generated from the JSON.
#                  Kept in sync after each review decision so the user can
#                  refresh their browser to see what's left.
#
# The review pass itself (run_review_pass) is the decision interface — runs in
# the terminal, walks pending items, and applies metadata via the same
# apply_metadata pipeline used by the main run.
# ---------------------------------------------------------------------------
def _review_json_path(folder):
    return os.path.join(folder, "review.json")

def _review_html_path(folder):
    return os.path.join(folder, "review.html")

def _load_review_json(folder):
    """Load existing review.json or return None."""
    path = _review_json_path(folder)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        console.print(f"   [yellow]⚠ Could not read existing review.json: {e}[/yellow]")
        return None

def _save_review_json(folder, data):
    """Save review.json atomically (write-then-rename) so a Ctrl-C mid-write
    doesn't leave a corrupted file."""
    path = _review_json_path(folder)
    tmp = path + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        console.print(f"   [yellow]⚠ Could not save review.json: {e}[/yellow]")
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

def _thumbnail_data_uri(image_path, max_dim=240):
    """Return a base64 data URI for a small JPEG thumbnail of `image_path`,
    or None if the image can't be opened. Used to embed thumbnails directly
    in review.html so the HTML is self-contained and portable."""
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in RAW_EXTENSIONS:
            try:
                import rawpy
                with rawpy.imread(image_path) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
                img = Image.fromarray(rgb)
            except ImportError:
                return None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = Image.open(image_path)
                img.load()
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        # Use thumbnail() which preserves aspect ratio in place
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None

def write_review_report(folder, review_queue):
    """Write review.json + review.html for the given queue.

    `review_queue` is a list of dicts. Each dict is enriched here with a
    stable id, a status (pending/applied/skipped), and a thumbnail data URI.
    If a review.json already exists, statuses are merged so previous decisions
    are preserved across re-runs.
    """
    if not review_queue:
        return

    # Merge with any existing decisions on disk (resume support)
    existing = _load_review_json(folder) or {"items": []}
    existing_by_path = {item.get("path"): item for item in existing.get("items", [])}

    enriched = []
    for q in review_queue:
        path = q.get("path", "")
        prev = existing_by_path.get(path)
        # Stable id from the path — survives reordering and re-runs
        item_id = prev.get("id") if prev else f"item_{abs(hash(path)) % 10**10:010d}"
        status = prev.get("status") if prev else "pending"
        decided_date = prev.get("decided_date") if prev else None

        thumb = prev.get("thumb") if prev and prev.get("thumb") else _thumbnail_data_uri(path)

        enriched.append({
            "id": item_id,
            "status": status,                       # pending | applied | skipped
            "decided_date": decided_date,           # set when status=applied
            "folder": q.get("folder", ""),
            "file": q.get("file", ""),
            "path": path,
            "raw_guess": q.get("raw_guess", ""),
            "found_date": q.get("found_date", ""),  # the AI's parsed date if any
            "confidence": q.get("confidence", 0),
            "comment": q.get("comment", "") or "",
            # Full metadata record so the review pass can rewrite without re-analyzing
            "tags": q.get("tags") or "",
            "raw_date": q.get("raw_date") or "",
            "gps": q.get("gps"),
            "scene": q.get("scene") or "",
            "setting": q.get("setting") or "",
            "flash": q.get("flash") or "",
            "thumb": thumb,
        })

    data = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": folder,
        "items": enriched,
    }
    _save_review_json(folder, data)
    _render_review_html(folder, data)

    pending_count = sum(1 for i in enriched if i["status"] == "pending")
    console.print(f"\n[cyan]📋 Review report:[/cyan] {_review_html_path(folder)}")
    if pending_count != len(enriched):
        console.print(f"   [dim]{pending_count} pending, {len(enriched) - pending_count} previously decided.[/dim]")

def _render_review_html(folder, data):
    """Generate the dark-mode HTML view from the review.json data."""
    items = data.get("items", [])
    pending = [i for i in items if i["status"] == "pending"]
    applied = [i for i in items if i["status"] == "applied"]
    skipped = [i for i in items if i["status"] == "skipped"]

    def card(item):
        is_done = item["status"] != "pending"
        status_class = item["status"]
        status_label = {
            "pending": "Pending review",
            "applied": f"Applied: {item.get('decided_date', '')}",
            "skipped": "Skipped permanently",
        }[item["status"]]

        thumb_html = (
            f'<img src="{item["thumb"]}" alt="thumbnail" loading="lazy">'
            if item.get("thumb")
            else '<div class="thumb-missing">No preview</div>'
        )

        comment_html = ""
        if item.get("comment") and item["comment"] != "—":
            comment_html = f'<div class="meta-row"><span class="label">Note:</span> <span>{_xml_escape(item["comment"])}</span></div>'

        scene_html = ""
        if item.get("scene"):
            scene_html = f'<div class="meta-row scene">{_xml_escape(item["scene"])}</div>'

        return f"""
    <div class="card {status_class}" data-id="{item['id']}">
      <div class="thumb">{thumb_html}</div>
      <div class="body">
        <div class="filename">{_xml_escape(item['file'])}</div>
        <div class="folder">{_xml_escape(item.get('folder', ''))}</div>
        {scene_html}
        <div class="meta-row"><span class="label">AI guess:</span> <span class="guess">{_xml_escape(item['raw_guess'])}</span></div>
        <div class="meta-row"><span class="label">Confidence:</span> <span class="conf">{item['confidence']}/10</span></div>
        {comment_html}
        <div class="status-pill {status_class}">{_xml_escape(status_label)}</div>
      </div>
    </div>"""

    cards_html = "\n".join(card(item) for item in items)
    generated = data.get("generated", "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Metadata-AI — Review Queue</title>
  <style>
    :root {{
      --bg: #0f1115;
      --panel: #181b22;
      --panel-2: #1f232c;
      --border: #2a2f3a;
      --text: #e6e8ec;
      --text-dim: #9aa3b2;
      --accent: #5b9dff;
      --green: #5dd39e;
      --amber: #f5a76b;
      --grey: #6b7280;
      --shadow: 0 4px 16px rgba(0,0,0,0.35);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }}
    body {{ padding: 32px 40px 80px; max-width: 1400px; margin: 0 auto; }}
    header {{ margin-bottom: 24px; }}
    h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }}
    .subtitle {{ color: var(--text-dim); font-size: 14px; }}
    .stats {{ display: flex; gap: 16px; margin: 20px 0 28px; flex-wrap: wrap; }}
    .stat {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 18px; min-width: 110px; }}
    .stat .num {{ font-size: 22px; font-weight: 600; }}
    .stat .lbl {{ font-size: 12px; color: var(--text-dim); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .stat.pending .num {{ color: var(--amber); }}
    .stat.applied .num {{ color: var(--green); }}
    .stat.skipped .num {{ color: var(--grey); }}

    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; box-shadow: var(--shadow); display: flex; flex-direction: column; transition: opacity 0.2s, border-color 0.2s; }}
    .card.applied {{ opacity: 0.55; border-color: rgba(93, 211, 158, 0.25); }}
    .card.skipped {{ opacity: 0.45; border-color: rgba(107, 114, 128, 0.25); }}
    .thumb {{ background: var(--panel-2); aspect-ratio: 4 / 3; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
    .thumb img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .thumb-missing {{ color: var(--text-dim); font-size: 13px; }}
    .body {{ padding: 14px 16px 16px; flex: 1; display: flex; flex-direction: column; gap: 6px; }}
    .filename {{ font-size: 14px; font-weight: 600; word-break: break-all; }}
    .folder {{ font-size: 12px; color: var(--text-dim); margin-bottom: 6px; }}
    .scene {{ font-size: 13px; color: var(--text-dim); font-style: italic; line-height: 1.4; padding-bottom: 4px; }}
    .meta-row {{ font-size: 13px; line-height: 1.5; }}
    .meta-row .label {{ color: var(--text-dim); }}
    .meta-row .guess {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
    .meta-row .conf {{ font-weight: 600; color: var(--amber); }}
    .status-pill {{ display: inline-block; align-self: flex-start; margin-top: 8px; padding: 3px 10px; border-radius: 999px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
    .status-pill.pending {{ background: rgba(245, 167, 107, 0.15); color: var(--amber); border: 1px solid rgba(245, 167, 107, 0.3); }}
    .status-pill.applied {{ background: rgba(93, 211, 158, 0.15); color: var(--green); border: 1px solid rgba(93, 211, 158, 0.3); }}
    .status-pill.skipped {{ background: rgba(107, 114, 128, 0.15); color: var(--text-dim); border: 1px solid rgba(107, 114, 128, 0.3); }}

    .help {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; margin-bottom: 24px; font-size: 13px; color: var(--text-dim); line-height: 1.6; }}
    .help code {{ background: var(--panel-2); padding: 2px 6px; border-radius: 4px; color: var(--text); font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; }}

    @media (max-width: 600px) {{
      body {{ padding: 20px 16px 60px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Metadata-AI — Review Queue</h1>
    <div class="subtitle">Generated {_xml_escape(generated)}</div>
  </header>

  <div class="stats">
    <div class="stat pending"><div class="num">{len(pending)}</div><div class="lbl">Pending</div></div>
    <div class="stat applied"><div class="num">{len(applied)}</div><div class="lbl">Applied</div></div>
    <div class="stat skipped"><div class="num">{len(skipped)}</div><div class="lbl">Skipped</div></div>
  </div>

  <div class="help">
    Run <code>python metadata-ai.py {_xml_escape(shlex.quote(folder))} --review</code> in your terminal to step through pending items.
    Refresh this page to see updated statuses after decisions are applied.
  </div>

  <div class="grid">{cards_html}
  </div>
</body>
</html>"""

    with open(_review_html_path(folder), "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Interactive review pass
# ---------------------------------------------------------------------------
def run_review_pass(folder, xmp_only=False):
    """Walk pending items in review.json and prompt the user for a decision.

    Decisions: [a]ccept the AI's date, [e]dit the date, [s]kip permanently, [q]uit.
    Each decision is persisted before moving on, so Ctrl-C is safe.
    On accept/edit, metadata is written via apply_metadata using the full
    record stored in review.json (no re-analysis needed).
    """
    data = _load_review_json(folder)
    if not data or not data.get("items"):
        console.print("[yellow]No review.json found in this folder — nothing to review.[/yellow]")
        return

    items = data["items"]
    pending = [i for i in items if i["status"] == "pending"]
    if not pending:
        console.print(f"[dim]All {len(items)} item(s) in review.json have already been decided.[/dim]")
        return

    total = len(pending)
    console.print(Panel.fit(
        f"[bold]Interactive Review[/bold] — [cyan]{total}[/cyan] pending photo(s)\n"
        f"[dim]Open [cyan]review.html[/cyan] in your browser to see thumbnails alongside this prompt.[/dim]\n\n"
        f"  [bold cyan]a[/bold cyan]  Accept the AI's date and write metadata\n"
        f"  [bold cyan]e[/bold cyan]  Enter a different date and write metadata\n"
        f"  [bold cyan]s[/bold cyan]  Skip permanently (no metadata written)\n"
        f"  [bold cyan]q[/bold cyan]  Quit (decisions so far are saved; resume with --review)",
        border_style="cyan",
        title="[bold]Metadata-AI[/bold]",
    ))

    decided_count = 0
    for idx, item in enumerate(pending, 1):
        console.print(f"\n[bold dim]─────────────────────────────────────────────────────[/bold dim]")
        console.print(f"[bold][{idx}/{total}][/bold] [cyan]{item['file']}[/cyan]  [dim]{item.get('folder', '')}[/dim]")
        if item.get("scene"):
            console.print(f"  [dim]Scene:[/dim]      {item['scene']}")
        console.print(
            f"  [dim]AI guess:[/dim]   [bold]{item['raw_guess']}[/bold]  "
            f"[yellow]({item['confidence']}/10 confidence)[/yellow]"
        )
        if item.get("comment") and item["comment"] not in ("—", ""):
            console.print(f"  [dim]Note:[/dim]       {item['comment']}")

        # Use questionary if available for a nicer select; fall back to text input
        choice = None
        try:
            import questionary
            if sys.stdin.isatty():
                choice = questionary.select(
                    "Decision:",
                    choices=[
                        questionary.Choice("Accept AI's date", value="a"),
                        questionary.Choice("Enter different date", value="e"),
                        questionary.Choice("Skip permanently", value="s"),
                        questionary.Choice("Quit (save progress)", value="q"),
                    ],
                ).ask()
        except ImportError:
            pass

        if choice is None:
            while True:
                try:
                    choice = console.input("  [bold]Decision[/bold] [[bold cyan]a[/bold cyan]/e/s/q]: ").strip().lower() or "a"
                except (EOFError, KeyboardInterrupt):
                    choice = "q"
                if choice in ("a", "e", "s", "q"):
                    break
                console.print("  [yellow]Please enter a, e, s, or q.[/yellow]")

        if choice == "q":
            console.print(f"\n[dim]Quitting. {decided_count}/{total} decided this session.[/dim]")
            console.print(f"[dim]Resume with:[/dim] [cyan]python metadata-ai.py {shlex.quote(folder)} --review[/cyan]")
            return

        if choice == "s":
            item["status"] = "skipped"
            _save_review_json(folder, data)
            _render_review_html(folder, data)
            console.print(f"  [dim]→ Skipped permanently.[/dim]")
            decided_count += 1
            continue

        if choice == "a":
            date_to_write = item.get("found_date") or item["raw_guess"]
            if not re.match(r'^\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}$', date_to_write):
                parsed, _ = parse_fuzzy_date(date_to_write)
                if not parsed:
                    console.print(f"  [yellow]⚠ Could not parse '{date_to_write}' as a date — use [e] to enter one manually.[/yellow]")
                    continue
                date_to_write = parsed
        else:  # choice == "e"
            try:
                user_date = console.input("  [bold]Date[/bold] [dim](YYYY:MM:DD, YYYY, 'circa 1970s', …)[/dim]: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not user_date:
                console.print("  [yellow]⚠ Empty input — leaving as pending.[/yellow]")
                continue
            parsed, _ = parse_fuzzy_date(user_date)
            if not parsed:
                console.print(f"  [yellow]⚠ Could not parse '{user_date}' as a date — leaving as pending.[/yellow]")
                continue
            date_to_write = parsed

        try:
            gps = tuple(item["gps"]) if item.get("gps") else None
            apply_metadata(
                item["path"], date_to_write,
                tags=item.get("tags") or None,
                comment=item.get("comment") if item.get("comment") not in ("—", "") else None,
                raw_date=item.get("raw_date") or None,
                gps=gps,
                xmp_only=xmp_only,
                scene=item.get("scene") or None,
                setting=item.get("setting") or None,
                flash=item.get("flash") or None,
            )
            item["status"] = "applied"
            item["decided_date"] = date_to_write
            _save_review_json(folder, data)
            _render_review_html(folder, data)
            decided_count += 1
        except Exception as e:
            console.print(f"  [red]✗ Write failed: {e} — leaving as pending.[/red]")

    remaining = sum(1 for i in items if i["status"] == "pending")
    console.print(f"\n[bold dim]─────────────────────────────────────────────────────[/bold dim]")
    console.print(
        f"[bold green]✓ Review complete:[/bold green] "
        f"[cyan]{decided_count}[/cyan] decided this session"
        + (f", [yellow]{remaining}[/yellow] still pending" if remaining else "")
    )
    console.print(f"[dim]View:[/dim] {_review_html_path(folder)}")

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
        console.print("      [dim]geopy not installed — skipping GPS tagging.[/dim]")
    except Exception as e:
        console.print(f"      [yellow]⚠ Geotagging error: {e}[/yellow]")
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
    """Convert a natural language time estimate to an hour (0-23).

    Returns None when the input contains no recognizable time signal — the
    caller decides what default to use. Keeping "couldn't parse" distinct
    from "noon" lets us tell the difference between an explicit midday photo
    and one where we just don't know.
    """
    if not text:
        return None
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
    # 24-hour clock like "14:00"
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
    return None  # nothing parseable — caller falls back to noon

def _process_folder(folder, files, cutoff_year, confidence_threshold, xmp_only, enable_geo, global_offset=0, global_total=None, folder_consensus=False, root_folder=None, dry_run=False, skip_dated=False, review_queue_accumulator=None, shared_durations=None, batch_totals=None):
    """Process one folder of photos.

    review_queue_accumulator: optional shared list. If provided, low-confidence
    items are appended to it instead of (or in addition to) being written to a
    folder-local report. Used by recursive mode to gather everything into a
    single review.html at the run root.

    shared_durations: optional deque passed from process_archive in recursive
    mode so the rolling ETA is computed across all folders, not reset per folder.

    batch_totals: optional dict with keys 'written', 'reviewed', 'no_date',
    'cutoff_skip', 'backs' for accumulating run-wide counts shown in the final
    summary panel.
    """
    processed_files = set()
    review_queue = []
    results = []  # collects dicts for deferred consensus write
    completed_paths = _load_checkpoint(root_folder or folder) if not dry_run else set()
    if dry_run and os.path.exists(_checkpoint_path(root_folder or folder)):
        console.print("   [dim]ℹ Dry-run: ignoring existing checkpoint, re-analyzing all files.[/dim]")
    no_date_count = 0
    cutoff_skip_count = 0

    folder_name = os.path.basename(folder)
    folder_name_useful = is_meaningful_folder_name(folder_name)
    hint = "" if folder_name_useful else "  [dim](low-signal name — not passed to VLM)[/dim]"
    console.print(f"   [bold]📁 {folder_name}[/bold]{hint}")
    console.print(f"   [dim]Archiving {len(files)} photo(s)…[/dim]\n")

    # Rolling-window ETA. In recursive mode, shared_durations is passed in from
    # process_archive so timings accumulate across all folders — giving a
    # meaningful estimate even for small folders and keeping it stable as the
    # run progresses. In single-folder mode a local deque is used instead.
    WINDOW_SIZE = 5
    MIN_SAMPLES = 3
    _durations: deque = shared_durations if shared_durations is not None else deque(maxlen=WINDOW_SIZE)
    _eta_str = ""

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[eta]}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Analyzing", total=len(files), eta="")

        for i in range(len(files)):
            current_file = files[i]
            if current_file in processed_files:
                progress.advance(task)
                continue

            current_path = os.path.join(folder, current_file)
            found_date = None
            found_comment = None
            raw_date_text = None
            confidence = 10
            gps_coords = None

            pos = global_offset + i + 1
            total_str = str(global_total) if global_total else str(len(files))

            _photo_start = time.monotonic()

            progress.update(task, description=f"[bold cyan]{current_file}[/bold cyan]")

            if current_path in completed_paths:
                console.print(f"  [dim][{pos}/{total_str}] Already processed — skipping: {current_file}[/dim]")
                progress.advance(task)
                continue

            if skip_dated and _has_existing_date(current_path):
                console.print(f"  [dim][{pos}/{total_str}] Already dated — skipping: {current_file}[/dim]")
                progress.advance(task)
                continue

            console.print(f"\n  [bold][{pos}/{total_str}][/bold] [cyan]{current_file}[/cyan]")

            # Step 1: Check if the NEXT photo is the back
            console.print(f"   [dim]1) Back-of-photo check…[/dim]")
            next_readable = False
            next_path = None
            if i + 1 < len(files):
                next_file = files[i+1]
                next_path = os.path.join(folder, next_file)

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _test = Image.open(next_path)
                        _test.load()
                        _test.close()
                    next_readable = True
                except Exception:
                    console.print(f"      [yellow]⚠ Could not open {next_file} — skipping back check.[/yellow]")

            if next_readable:
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
                back_resp = ask_vlm(next_path, back_prompt)
                is_back = False
                if back_resp:
                    is_back_line = re.search(r'IS_BACK:\s*([^\n]+)', back_resp)
                    if is_back_line and is_back_line.group(1).strip().lower().startswith("yes"):
                        is_back = True

                if is_back:
                    console.print(f"      [green]Back confirmed:[/green] {next_file}")
                    processed_files.add(next_file)

                    date_line = re.search(r'DATE:\s*([^\n]+)', back_resp)
                    comment_line = re.search(r'COMMENT:\s*([^\n]+)', back_resp)
                    raw_date_str = date_line.group(1).strip() if date_line else "none"
                    raw_comment = comment_line.group(1).strip() if comment_line else "none"

                    ocr_context = run_tesseract(next_path)
                    if ocr_context and raw_date_str.lower() == "none":
                        console.print(f"      [dim]Tesseract OCR found text — re-checking for date…[/dim]")
                        ocr_prompt = (
                            f"This is the back of a photo. OCR extracted: {ocr_context}\n"
                            "Extract any date from this text in YYYY:MM:DD format, or 'circa 1950s', etc. "
                            "If no date, return 'none'."
                        )
                        raw_date_str = ask_vlm(next_path, ocr_prompt).strip()

                    found_date, raw_date_text = parse_fuzzy_date(raw_date_str)
                    if found_date:
                        fuzzy_note = f" [dim](fuzzy: {raw_date_text})[/dim]" if raw_date_text else ""
                        console.print(f"      [dim]Date from back:[/dim] [bold]{found_date}[/bold]{fuzzy_note}")
                    else:
                        console.print(f"      [dim]No date on back — falling through to VLM guess.[/dim]")

                    if raw_comment.lower() != "none" and raw_comment:
                        found_comment = raw_comment
                        console.print(f"      [dim]Comment:[/dim] {found_comment}")
                    else:
                        console.print(f"      [dim]No comment on back.[/dim]")
                else:
                    console.print(f"      [dim]No back detected.[/dim]")
            else:
                console.print(f"      [dim]No next image to check.[/dim]")

            # Step 2: Check IPTC keywords for a date
            if not found_date:
                console.print(f"   [dim]2) IPTC keyword check…[/dim]")
                iptc_keywords = get_iptc_metadata(current_path)
                if iptc_keywords:
                    for keyword in iptc_keywords:
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
                            fuzzy_note = f" [dim](fuzzy: {raw_date_text})[/dim]" if raw_date_text else ""
                            console.print(f"      [dim]Date from IPTC '[/dim][bold]{keyword}[/bold][dim]':[/dim] [bold]{found_date}[/bold]{fuzzy_note}")
                            break
                    if not found_date:
                        console.print(f"      [dim]No date found in IPTC keywords.[/dim]")
                else:
                    console.print(f"      [dim]No IPTC keywords found.[/dim]")

            # Step 3: VLM analysis. Internally this is two focused calls when needed:
            # a date-only call (only when no date was found from back-of-photo or IPTC)
            # followed by a description call. Combining the two tasks measurably hurt
            # confidence calibration, so they're separated. From the user's point of
            # view this is one "analyze the image" step — the calls aren't surfaced.
            raw_time = None
            vlm_scene = None
            vlm_setting = None
            vlm_flash = None
            tags_resp = None
            geo_resp_inline = None

            # Only inject the folder name as VLM context when it looks meaningful —
            # noise like "New Folder (2)" or "Scans_Batch_3" otherwise primes the
            # model with garbage.
            folder_hint_date = (
                f"The folder containing this photo is named '{folder_name}' — treat this as high-confidence information for the date. "
                if folder_name_useful else ""
            )
            folder_hint_loc = (
                f"The folder containing this photo is named '{folder_name}' — treat this as high-confidence information for the location. "
                if folder_name_useful else ""
            )

            console.print(f"   [dim]3) Analyzing image…[/dim]")

            # Internal date call — focused single-task prompt for date estimation.
            # Skipped when a date was already extracted from back-of-photo or IPTC.
            if not found_date:
                date_prompt = (
                    "Analyze the fashion, hairstyles, technology, and setting in this photo to estimate when it was taken. "
                    f"{folder_hint_date}"
                    "Estimate the date as specifically as possible — could be YYYY:MM:DD, YYYY:MM, YYYY, "
                    "a decade like '1970s', or 'circa 1965'. "
                    f"The date must be before {cutoff_year}. "
                    "Also provide a confidence score from 1-10 for your date estimate.\n"
                    "Reply in EXACTLY this format with no other text:\n"
                    "DATE: <your estimate>\n"
                    "CONFIDENCE: <score 1-10>"
                )
                with console.status("[dim]Dating…[/dim]", spinner="dots"):
                    date_resp = ask_vlm(current_path, date_prompt, max_tokens=200)
                if not date_resp:
                    console.print(f"      [yellow]⚠ VLM returned no response for date estimate.[/yellow]")
                    date_resp = ""

                date_line = re.search(r'DATE:\s*([^\n]+)',    date_resp)
                conf_line = re.search(r'CONFIDENCE:\s*(\d+)', date_resp)

                if date_line:
                    raw_guess = date_line.group(1).strip()
                    confidence = int(conf_line.group(1)) if conf_line else 5
                    found_date, raw_date_text = parse_fuzzy_date(raw_guess)
                    if found_date:
                        parts = found_date.split(':')
                        try:
                            year_val = int(parts[0])
                            month_val = int(parts[1])
                            day_val = int(parts[2].split()[0])
                            if not (MIN_PHOTO_YEAR <= year_val <= MAX_YEAR and 1 <= month_val <= 12 and 1 <= day_val <= 31):
                                console.print(f"      [yellow]⚠ Invalid date from VLM ('{raw_guess}') — discarding.[/yellow]")
                                found_date = None
                            else:
                                fuzzy_note = f" [dim]— fuzzy: {raw_date_text}[/dim]" if raw_date_text else ""
                                conf_color = "green" if confidence >= 7 else "yellow" if confidence >= 5 else "red"
                                console.print(
                                    f"      [dim]Date:[/dim]       [bold]{found_date}[/bold]  "
                                    f"[[{conf_color}]{confidence}/10[/{conf_color}]]{fuzzy_note}"
                                )
                        except (IndexError, ValueError):
                            console.print(f"      [yellow]⚠ Invalid date format ('{raw_guess}') — discarding.[/yellow]")
                            found_date = None
                    else:
                        console.print(f"      [dim]VLM could not determine a date. Raw: '[/dim]{raw_guess}[dim]' ({confidence}/10)[/dim]")

            # Internal description call — time, scene, setting, flash, location, keywords.
            # The location prompt explicitly requires a specific identifiable feature
            # (landmark, sign, distinctive building) — not just general landscape appearance.
            # We also ask for a location confidence score and only geocode high-confidence hits.
            geo_instruction = (
                "LOCATION: <name a specific city, landmark, or building ONLY if you can identify it "
                "from a visible sign, recognisable landmark, or distinctive architecture — "
                "do NOT guess from general landscape appearance such as mountains or forest; "
                "if uncertain write 'none'>\n"
                "LOCATION_CONFIDENCE: <1-10 — how certain are you? "
                "9-10: unmistakable landmark or sign visible; "
                "6-8: strong architectural or environmental cues; "
                "1-5: mostly guessing from general appearance>\n"
            ) if enable_geo else ""

            location_context = folder_hint_loc if enable_geo else ""

            desc_prompt = (
                f"{location_context}"
                "Describe this photo. Reply in EXACTLY this format with no other text:\n"
                "TIME: <time of day — e.g. 'morning', 'midday', 'afternoon', 'evening', or '3pm'>\n"
                "SCENE: <one sentence describing the scene>\n"
                "SETTING: <'indoor' or 'outdoor'>\n"
                "FLASH: <'yes' or 'no' — whether flash appears to have fired>\n"
                f"{geo_instruction}"
                "KEYWORDS: <5 descriptive keywords, comma separated>"
            )

            with console.status("[dim]Describing…[/dim]", spinner="dots"):
                resp = ask_vlm(current_path, desc_prompt)
            if not resp:
                console.print(f"      [yellow]⚠ VLM returned no response — skipping description.[/yellow]")
                resp = ""

            # Parse description fields from the response
            time_line      = re.search(r'TIME:\s*([^\n]+)',                resp)
            scene_line     = re.search(r'SCENE:\s*([^\n]+)',               resp)
            setting_line   = re.search(r'SETTING:\s*([^\n]+)',             resp)
            flash_line     = re.search(r'FLASH:\s*([^\n]+)',               resp)
            geo_line       = re.search(r'LOCATION:\s*([^\n]+)',            resp) if enable_geo else None
            geo_conf_line  = re.search(r'LOCATION_CONFIDENCE:\s*(\d+)',   resp) if enable_geo else None
            keywords_line  = re.search(r'KEYWORDS:\s*([^\n]+)',            resp)

            geo_confidence = int(geo_conf_line.group(1)) if geo_conf_line else 0

            # Parse the time field first, then use the parse result as the validity signal.
            # This is more robust than denylisting phrases like "studio lighting" — if the
            # VLM returned "early morning, around 7am" (>40 chars, used to be discarded),
            # the parser still finds 7am. If it returned a hedge like "cannot determine
            # from indoor lighting", the parser finds nothing and the caller falls back.
            _raw_time_str = _clean_vlm_field(time_line.group(1)) if time_line else None
            time_hour = _parse_time_of_day(_raw_time_str)
            raw_time = _raw_time_str if time_hour is not None else None

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
            geo_resp_inline     = _clean_vlm_field(geo_line.group(1))  if geo_line  else None
            geo_resp_confidence = geo_confidence

            if raw_time:    console.print(f"      [dim]Time:[/dim]       {raw_time}")
            if vlm_scene:   console.print(f"      [dim]Scene:[/dim]      {vlm_scene}")
            if vlm_setting: console.print(f"      [dim]Setting:[/dim]    {vlm_setting}")
            if vlm_flash:   console.print(f"      [dim]Flash:[/dim]      {vlm_flash}")
            if tags_resp:   console.print(f"      [dim]Keywords:[/dim]   [dim]{tags_resp}[/dim]")

            if found_date:
                applied_hour = time_hour if time_hour is not None else 12
                found_date = found_date[:11] + f"{applied_hour:02d}:00:00"
                time_note = f" [dim](~{raw_time})[/dim]" if raw_time else ""
                console.print(f"      [dim]Timestamp:[/dim]  [bold]{found_date}[/bold]{time_note}")

            if enable_geo:
                console.print(f"   [dim]4) Location check…[/dim]")
                geo_resp = geo_resp_inline or ""
                geo_resp = re.sub(r'\s*\(.*?\)', '', geo_resp).strip()
                loc_conf = geo_resp_confidence

                # Gate 1: basic validity — not empty, not a hedge phrase
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
                # Gate 2: confidence threshold — only geocode when the VLM is sure
                # it saw a specific identifiable feature, not just guessed from landscape.
                GEO_CONFIDENCE_THRESHOLD = 7
                if is_valid_location and loc_conf < GEO_CONFIDENCE_THRESHOLD:
                    console.print(
                        f"      [dim]Location skipped:[/dim] [yellow]{geo_resp}[/yellow] "
                        f"[dim](confidence {loc_conf}/10 < {GEO_CONFIDENCE_THRESHOLD} — landscape guess)[/dim]"
                    )
                    is_valid_location = False

                if is_valid_location:
                    console.print(f"      [dim]Location:[/dim]   {geo_resp} [dim]({loc_conf}/10)[/dim]")
                    gps_coords = geolocate(geo_resp)
                    if gps_coords:
                        console.print(f"      [dim]GPS:[/dim]        {gps_coords[0]:.4f}, {gps_coords[1]:.4f}")
                    else:
                        simplified = geo_resp.split(',')[0].strip()
                        if simplified and simplified.lower() != geo_resp.lower():
                            console.print(f"      [dim]Retrying with:[/dim] '{simplified}'")
                            gps_coords = geolocate(simplified)
                            if gps_coords:
                                console.print(f"      [dim]GPS:[/dim]        {gps_coords[0]:.4f}, {gps_coords[1]:.4f}")
                            else:
                                console.print(f"      [dim]Could not resolve GPS — skipping.[/dim]")
                        else:
                            console.print(f"      [dim]Could not resolve GPS — skipping.[/dim]")
                else:
                    if not geo_resp or geo_resp.lower() == "none":
                        console.print(f"      [dim]No location identified.[/dim]")

            console.print(f"   [dim]5) Writing metadata…[/dim]")
            if found_date:
                try:
                    year = int(found_date[:4])
                    if year < cutoff_year:
                        if folder_consensus:
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
                            console.print(f"      [dim]Queued for consensus write.[/dim]")
                        else:
                            if confidence >= confidence_threshold:
                                if dry_run:
                                    console.print(f"      [dim][DRY RUN] Would write: {found_date} | tags: {tags_resp}[/dim]")
                                else:
                                    apply_metadata(current_path, found_date, tags=tags_resp, comment=found_comment,
                                                   raw_date=raw_date_text, gps=gps_coords, xmp_only=xmp_only,
                                                   scene=vlm_scene, setting=vlm_setting, flash=vlm_flash)
                                    _save_checkpoint(root_folder or folder, current_path)
                            else:
                                console.print(f"      [yellow]⚠ Low confidence ({confidence}/10) — added to review queue.[/yellow]")
                                review_queue.append({
                                    "folder": folder_name,
                                    "file": current_file,
                                    "path": current_path,
                                    "raw_guess": raw_date_text or found_date,
                                    "found_date": found_date,
                                    "confidence": confidence,
                                    "comment": found_comment or "—",
                                    "tags": tags_resp,
                                    "raw_date": raw_date_text,
                                    "gps": list(gps_coords) if gps_coords else None,
                                    "scene": vlm_scene,
                                    "setting": vlm_setting,
                                    "flash": vlm_flash,
                                })
                    else:
                        console.print(f"      [dim]⏭ Skipping: date {year} is {cutoff_year} or later.[/dim]")
                        cutoff_skip_count += 1
                except Exception as e:
                    console.print(f"      [red]✗ Write error: {e}[/red]")
            else:
                has_something = any([tags_resp, vlm_scene, vlm_setting, vlm_flash, found_comment, gps_coords])
                if has_something:
                    console.print(f"      [yellow]⚠ No date found — writing description/keywords only.[/yellow]")
                    if dry_run:
                        console.print(f"      [dim][DRY RUN] Would write: tags: {tags_resp} | scene: {vlm_scene}[/dim]")
                    else:
                        apply_metadata(current_path, None, tags=tags_resp, comment=found_comment,
                                       raw_date=raw_date_text, gps=gps_coords, xmp_only=xmp_only,
                                       scene=vlm_scene, setting=vlm_setting, flash=vlm_flash)
                        _save_checkpoint(root_folder or folder, current_path)
                else:
                    console.print(f"      [red]✗ No date or description found — skipping.[/red]")
                no_date_count += 1

            # Record this photo's wall-clock duration and recompute ETA.
            # In recursive mode global_total is set, so remaining spans the
            # whole batch — not just the current folder.
            _durations.append(time.monotonic() - _photo_start)
            photos_done  = global_offset + i + 1
            photos_total = global_total if global_total is not None else len(files)
            remaining    = max(0, photos_total - photos_done)
            if len(_durations) >= MIN_SAMPLES and remaining > 0:
                avg_sec  = sum(_durations) / len(_durations)
                eta_sec  = avg_sec * remaining
                _eta_str = f"[dim]{_format_eta(eta_sec)} remaining[/dim]"
            elif remaining == 0:
                _eta_str = ""
            progress.update(task, eta=_eta_str)
            progress.advance(task)

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
            console.print(f"\n   [cyan]🗳 Consensus year: {consensus_year}[/cyan] [dim]({len(high_conf_years)} high-confidence result(s))[/dim]")
        else:
            consensus_year = None
            console.print(f"\n   [yellow]⚠ No high-confidence results to derive consensus year.[/yellow]")

        for r in results:
            date = r['found_date']
            if consensus_year and r['confidence'] < confidence_threshold:
                date = f"{consensus_year}:{date[5:]}"
                console.print(f"      [dim]📅 {r['file']}: overridden to {date} via consensus[/dim]")
            if r['confidence'] >= confidence_threshold or consensus_year:
                if dry_run:
                    console.print(f"      [dim][DRY RUN] Would write: {date} | tags: {r['tags']}[/dim]")
                else:
                    apply_metadata(r['path'], date, tags=r['tags'], comment=r['comment'],
                                   raw_date=r['raw_date'], gps=r['gps'], xmp_only=xmp_only,
                                   scene=r['scene'], setting=r['setting'], flash=r['flash'])
                    _save_checkpoint(root_folder or folder, r['path'])
            else:
                review_queue.append({
                    "folder": folder_name,
                    "file": r['file'],
                    "path": r['path'],
                    "raw_guess": r['raw_date'] or date,
                    "found_date": date,
                    "confidence": r['confidence'],
                    "comment": r['comment'] or "—",
                    "tags": r.get('tags'),
                    "raw_date": r.get('raw_date'),
                    "gps": list(r['gps']) if r.get('gps') else None,
                    "scene": r.get('scene'),
                    "setting": r.get('setting'),
                    "flash": r.get('flash'),
                })

    if review_queue_accumulator is not None:
        review_queue_accumulator.extend(review_queue)
    else:
        write_review_report(folder, review_queue)

    # Folder summary
    total          = len(files)
    backs_consumed = len(processed_files)
    reviewed       = len(review_queue)
    written        = max(0, total - backs_consumed - reviewed - no_date_count - cutoff_skip_count)

    # Accumulate into the batch-wide counters when in recursive mode
    if batch_totals is not None:
        batch_totals['written']      += written
        batch_totals['reviewed']     += reviewed
        batch_totals['no_date']      += no_date_count
        batch_totals['cutoff_skip']  += cutoff_skip_count
        batch_totals['backs']        += backs_consumed
        batch_totals['scanned']      += total

    summary_lines = [f"[bold green]{written} written[/bold green]  of {total} scanned"]
    if backs_consumed:
        summary_lines.append(f"[dim]{backs_consumed} back-of-photo consumed[/dim]")
    if cutoff_skip_count:
        summary_lines.append(f"[dim]{cutoff_skip_count} skipped (after cutoff)[/dim]")
    if reviewed:
        summary_lines.append(f"[yellow]{reviewed} queued for review[/yellow]")
    if no_date_count:
        summary_lines.append(f"[dim]{no_date_count} no date found[/dim]")
    console.print(f"\n  {' · '.join(summary_lines)}")


def process_archive(folder, cutoff_year=2010, confidence_threshold=7, xmp_only=False, enable_geo=False, recursive=False, folder_consensus=False, dry_run=False, skip_dated=False):
    """Process a directory of photos. Returns the folder where review.json was
    written (or None if no review queue was produced), so the caller can offer
    an interactive review pass at end of run."""
    if not os.path.exists(folder):
        console.print(f"[red]Directory not found: {folder}[/red]")
        return None

    # Run summary panel
    flags = []
    if recursive:       flags.append("recursive")
    if enable_geo:      flags.append("geotag")
    if folder_consensus: flags.append("consensus")
    if xmp_only:        flags.append("xmp-only")
    if dry_run:         flags.append("DRY RUN")
    if skip_dated:      flags.append("skip-dated")
    flags_str = "  ".join(f"[cyan]{f}[/cyan]" for f in flags) if flags else "[dim]none[/dim]"
    console.print(Panel.fit(
        f"[bold]Metadata-AI[/bold]\n\n"
        f"[dim]Folder:[/dim]     {folder}\n"
        f"[dim]Model:[/dim]      {MODEL_ID}\n"
        f"[dim]Cutoff:[/dim]     before {cutoff_year}\n"
        f"[dim]Confidence:[/dim] ≥{confidence_threshold}/10\n"
        f"[dim]Options:[/dim]    {flags_str}",
        border_style="cyan",
        title="[bold]Metadata-AI[/bold]",
    ))

    if recursive:
        all_paths = []
        for root, _, filenames in os.walk(folder):
            for f in filenames:
                if f.lower().endswith(EXTENSIONS):
                    all_paths.append(os.path.join(root, f))
        all_paths = natsorted(all_paths)
        global_total = len(all_paths)
        console.print(f"\n[bold]📂 {global_total} files across all subfolders[/bold]")

        accumulated_review_queue = []
        shared_durations = deque(maxlen=5)
        batch_totals = {'scanned': 0, 'written': 0, 'reviewed': 0,
                        'no_date': 0, 'cutoff_skip': 0, 'backs': 0}

        global_offset = 0
        for subfolder, path_iter in groupby(all_paths, key=os.path.dirname):
            subfolder_files = [os.path.basename(p) for p in path_iter]
            console.print(f"\n[bold cyan]📁 {subfolder}[/bold cyan] [dim]({len(subfolder_files)} files)[/dim]")
            _process_folder(subfolder, subfolder_files, cutoff_year, confidence_threshold, xmp_only, enable_geo,
                            global_offset=global_offset, global_total=global_total, folder_consensus=folder_consensus,
                            root_folder=folder, dry_run=dry_run, skip_dated=skip_dated,
                            review_queue_accumulator=accumulated_review_queue,
                            shared_durations=shared_durations, batch_totals=batch_totals)
            global_offset += len(subfolder_files)

        # Single combined report at the run root.
        write_review_report(folder, accumulated_review_queue)

        # Batch summary panel
        bt = batch_totals
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_column(style="dim", width=16)
        tbl.add_column()
        tbl.add_row("Total scanned",  str(bt['scanned']))
        tbl.add_row("[green]Written[/green]",       f"[green]{bt['written']}[/green]")
        if bt['backs']:
            tbl.add_row("[dim]Backs consumed[/dim]", str(bt['backs']))
        if bt['cutoff_skip']:
            tbl.add_row("[dim]After cutoff[/dim]",   str(bt['cutoff_skip']))
        if bt['reviewed']:
            tbl.add_row("[yellow]Review queue[/yellow]",   f"[yellow]{bt['reviewed']}[/yellow]")
        if bt['no_date']:
            tbl.add_row("[dim]No date found[/dim]",  str(bt['no_date']))
        console.print(Panel(tbl, title="[bold]Run Complete — Batch Total[/bold]", border_style="cyan"))

        if not dry_run:
            _clear_checkpoint(folder)
        return folder if accumulated_review_queue else None

    files = natsorted([f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)])
    _process_folder(folder, files, cutoff_year, confidence_threshold, xmp_only, enable_geo, folder_consensus=folder_consensus, dry_run=dry_run, skip_dated=skip_dated)
    if not dry_run:
        _clear_checkpoint(folder)
    # Return the folder if a review.json was written (i.e. there were skipped items)
    return folder if os.path.exists(_review_json_path(folder)) else None


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
- TITLE: if the video has a clear formal title use it; for personal footage
  use a descriptive title like "1970s Family Footage" or "Summer 1965 Vacation"; otherwise none
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

    # Clamp interval so we always extract at least one frame.
    # A 9-second clip with interval=30 would produce zero frames and cause ffmpeg
    # to fail with -22 (Invalid argument / no packets written).
    effective_interval = interval
    if effective_interval >= duration:
        effective_interval = max(1, int(duration / 2))
        console.print(
            f"  [yellow]⚠ Interval ({interval}s) ≥ video duration ({duration:.1f}s) "
            f"— reduced to {effective_interval}s.[/yellow]"
        )

    pattern = os.path.join(out_dir, "frame_%08d.jpg")
    # Express fps as an exact fraction string so ffmpeg receives an unambiguous
    # rational value regardless of locale or float formatting.
    fps_str = f"1/{int(effective_interval)}"

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps_str}",
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
        ts = idx * effective_interval
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
    _looks_technical = bool(re.search(
        r'(fps|mbps|kbps|\d+x\d+|bitrate|codec|h264|h265|hevc|avc)',
        video_name, re.IGNORECASE))
    context_hint = (f"\nContext: this frame is from a video file named '{video_name}'. "
                    "Treat this as high-confidence date and location information.")\
                   if video_name and not _looks_technical else ""
    resp = ask_vlm(image_path, VIDEO_FRAME_PROMPT + context_hint)
    if not resp:
        return {"description": "", "date_raw": "unknown", "year": None, "confidence": 0}
    description = _video_parse_field(resp, "DESCRIPTION")
    date_raw    = _video_parse_field(resp, "DATE") or "unknown"
    conf_str    = _video_parse_field(resp, "CONFIDENCE")
    try:
        confidence = max(1, min(10, int(re.search(r'\d+', conf_str).group())))
    except Exception:
        confidence = 5
    year = _video_parse_year(date_raw)
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
        console.print(f"      [dim](Summarizing {len(selected)} of {len(all_descriptions)} frames to fit context window)[/dim]")
    frame_descriptions = "\n\n".join(selected)
    prompt = VIDEO_SUMMARY_PROMPT.format(interval=interval, frame_descriptions=frame_descriptions)
    with console.status("[dim]Synthesizing summary…[/dim]", spinner="dots"):
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL_ID,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            result = resp.choices[0].message.content.strip()

            finish_reason = getattr(resp.choices[0], 'finish_reason', None)
            ends_mid_sentence = result and not result.rstrip().endswith(('.', '!', '?', '"', ')', ']'))
            if (finish_reason == "length" or ends_mid_sentence) and len(selected) > 1:
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
                    console.print("[yellow]⚠ Summary may be incomplete (truncated after retry).[/yellow]")

            return result
        except Exception as e:
            console.print(f"[red]✗ Summary generation failed: {e}[/red]")
            return "(Summary generation failed.)"

def _video_extract_metadata(summary, consensus_yr):
    """Ask the VLM to extract structured metadata from the summary."""
    with console.status("[dim]Extracting metadata fields…[/dim]", spinner="dots"):
        prompt = VIDEO_METADATA_PROMPT.format(summary=summary)
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            console.print(f"[red]✗ Metadata extraction failed: {e}[/red]")
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
            console.print(f"   [yellow]⚠ ffmpeg error: {result.stderr[-300:]}[/yellow]")
            if os.path.exists(tmp_path): os.remove(tmp_path)
            return False
        os.replace(tmp_path, video_path)
        return True
    except Exception as e:
        console.print(f"   [yellow]⚠ ffmpeg exception: {e}[/yellow]")
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
        console.print(f"[red]Video file not found: {video_path}[/red]")
        return

    default_output = str(Path(video_path).parent / (Path(video_path).stem + "_summary.txt"))
    if output_path is None:
        output_path = _ask("Output summary file", default_output)

    console.print(Panel.fit(
        f"[bold]Metadata-AI — Video Analysis[/bold]\n\n"
        f"[dim]Video:[/dim]    {video_path}\n"
        f"[dim]Model:[/dim]    {MODEL_ID}\n"
        f"[dim]Interval:[/dim] every {interval}s\n"
        f"[dim]Output:[/dim]   {output_path}",
        border_style="cyan",
        title="[bold]Metadata-AI[/bold]",
    ))

    with tempfile.TemporaryDirectory() as tmp_dir:
        with console.status("[cyan]Extracting frames…[/cyan]", spinner="dots"):
            try:
                frames = _video_extract_frames(video_path, interval, tmp_dir)
            except Exception as e:
                console.print(f"   [red]✗ Frame extraction failed: {e}[/red]")
                return
        console.print(f"  [green]✓[/green] {len(frames)} frame(s) extracted")

        frame_analyses = []
        video_stem = Path(video_path).stem
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing frames", total=len(frames))
            for i, (ts, img_path) in enumerate(frames, 1):
                progress.update(task, description=f"[bold cyan]{_video_format_ts(ts)}[/bold cyan]")
                data = _video_analyze_frame(ts, img_path, i, len(frames), video_name=video_stem)
                frame_analyses.append((ts, data))
                conf_color = "green" if data["confidence"] >= 6 else "yellow"
                yr_str = str(data["year"]) if data["year"] else "?"
                flag = "✓" if data["confidence"] >= 6 and data["year"] else "~"
                progress.console.print(
                    f"  [{flag}] [dim]{_video_format_ts(ts)}[/dim]  "
                    f"{data['date_raw']:<20} → {yr_str:<6}  "
                    f"[[{conf_color}]{data['confidence']}/10[/{conf_color}]]"
                )
                progress.advance(task)

        year_conf_pairs = [(d["year"], d["confidence"]) for _, d in frame_analyses]
        cons_year, votes, eligible, majority = _video_consensus_year(year_conf_pairs, threshold=6)

        if cons_year:
            majority_note = "" if majority else "  [yellow](plurality only — low agreement)[/yellow]"
            console.print(f"\n  [bold]Consensus year:[/bold] [cyan]{cons_year}[/cyan]{majority_note}")
            console.print(f"  [dim]Votes: {votes}/{eligible} high-confidence frames[/dim]")
        else:
            console.print("\n  [yellow]No consensus date — not enough confident frame estimates.[/yellow]")

        summary  = _video_synthesize_summary(frame_analyses, interval)
        metadata = _video_extract_metadata(summary, cons_year)

    # Metadata preview panel
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for field in ["title", "description", "keywords", "date", "location", "genre", "artist"]:
        val = metadata.get(field) or "[dim](none)[/dim]"
        table.add_row(field.capitalize() + ":", val)
    summary_preview = " ".join(summary[:400].split()) + ("…" if len(summary) > 400 else "")
    table.add_row("Summary:", f"[dim]{summary_preview}[/dim]")
    console.print(Panel(table, title="[bold]Metadata Preview[/bold]", border_style="cyan"))

    if _yn("Edit metadata fields before writing?"):
        console.print("  [dim](Press Enter to keep current value)[/dim]")
        for field in ["title", "description", "keywords", "date", "location", "genre", "artist"]:
            new_val = _ask(f"  {field.capitalize()}", metadata.get(field) or "")
            if new_val:
                metadata[field] = new_val
        new_summary = _ask(f"  Summary ({len(summary)} chars — Enter to keep)", "")
        if new_summary:
            summary = new_summary

    if _yn("Write metadata to video file?"):
        with console.status(f"[cyan]Writing metadata to {os.path.basename(video_path)}…[/cyan]"):
            ok = _video_write_metadata(video_path, metadata, summary)
        if ok:
            console.print(f"[bold green]✓ Metadata written:[/bold green] {os.path.basename(video_path)}")
        else:
            console.print("[red]✗ Metadata write failed — original file unchanged.[/red]")
    else:
        console.print("[dim]Skipping metadata write.[/dim]")

    report = _video_build_report(video_path, interval, frame_analyses, summary, cons_year)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    console.print(f"\n[bold green]✓ Report saved:[/bold green] {output_path}")

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
    parser.add_argument("--review", action="store_true", default=False,
                        help="Run interactive review of pending items in <directory>/review.json (skips photo analysis)")
    parser.add_argument("--model", type=str, default=None,
                        help=f"LM Studio model ID to use (default: {MODEL_ID})")
    parser.add_argument("--video-interval", type=int, default=None, metavar="SECONDS",
                        help="Seconds between frames for video analysis (default: 30)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output summary .txt path for video analysis (default: <video>_summary.txt)")

    args = parser.parse_args()

    # Apply --model override globally so all VLM helpers see it.
    if args.model:
        MODEL_ID = args.model
        console.print(f"[dim]Using model: {MODEL_ID}[/dim]")

    # Directory or file — prompt if not provided
    if args.directory:
        input_path = re.sub(r'\\(.)', r'\1', args.directory.strip())
    else:
        input_path = re.sub(r'\\(.)', r'\1', _ask("Directory or file path", default=DIRECTORY)) or DIRECTORY

    # ── Review-only mode ────────────────────────────────────────────────────
    if args.review:
        if not os.path.isdir(input_path):
            console.print(f"[red]--review expects a directory containing review.json. Got: {input_path}[/red]")
            sys.exit(1)
        xmp_only = args.xmp_only or False
        run_review_pass(input_path, xmp_only=xmp_only)
        sys.exit(0)

    # If a file path was given, route directly to the right processor
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            if args.video_interval is not None:
                interval = args.video_interval
            else:
                raw = _ask("Seconds between frames", "30")
                try:    interval = int(raw)
                except: interval = 30
            output_path = getattr(args, 'output', None)
            process_video(input_path, interval, output_path=output_path)
        elif ext in EXTENSIONS:
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
            console.print(f"[red]Unsupported file type: {ext}[/red]")
        sys.exit(0)

    directory = input_path

    # ── Photo or video mode selection ─────────────────────────────────────────
    if _yn_select("Analyze video files in this directory?"):
        video_files = natsorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ])
        if not video_files:
            console.print("[dim]No video files found — switching to photo mode.[/dim]")
        else:
            raw = _ask("Seconds between frames", "30")
            try:    video_interval = int(raw)
            except: video_interval = 30
            if not args.model:
                model_input = _ask(f"LM Studio model", MODEL_ID)
                if model_input and model_input != MODEL_ID:
                    MODEL_ID = model_input
                    console.print(f"[dim]Using model: {MODEL_ID}[/dim]")
            console.print(f"\n[bold]{len(video_files)} video file(s) found.[/bold]")
            for vf in video_files:
                console.print(f"\n[bold cyan]📹 {vf}[/bold cyan]")
                process_video(os.path.join(directory, vf), video_interval)
            sys.exit(0)

    # ── Photo mode ───────────────────────────────────────────────────────────
    if args.cutoff is not None:
        cutoff_year = args.cutoff
    else:
        raw = _ask("Skip photos dated from which year or later?", "2010")
        try:    cutoff_year = int(raw)
        except:
            console.print("[dim]Invalid year — defaulting to 2010.[/dim]")
            cutoff_year = 2010

    if args.confidence is not None:
        confidence_threshold = args.confidence
    else:
        raw = _ask("Confidence threshold for auto-write (1-10)", "7")
        try:    confidence_threshold = int(raw)
        except: confidence_threshold = 7

    if args.xmp_only is not None:
        xmp_only = args.xmp_only
    else:
        xmp_only = _yn_select("Write metadata to XMP sidecar files only?")

    if args.geotag is not None:
        enable_geo = args.geotag
    else:
        enable_geo = _yn_select("Enable geotagging?")

    if args.recursive is not None:
        recursive = args.recursive
    else:
        recursive = _yn_select("Recursively process subfolders?")

    if args.consensus is not None:
        folder_consensus = args.consensus
    else:
        folder_consensus = _yn_select("Use folder consensus year for uncertain dates?")

    if args.dry_run:
        console.print("\n[bold yellow]⚠ DRY RUN MODE — no files will be modified.[/bold yellow]\n")

    if not args.dry_run:
        progress_file = _checkpoint_path(directory)
        if os.path.exists(progress_file):
            with open(progress_file) as pf:
                completed_count = sum(1 for line in pf if line.strip())
            if not _yn(f"Found a previous session with {completed_count} completed file(s). Resume?", default_yes=True):
                _clear_checkpoint(directory)
                console.print("[dim]Starting fresh.[/dim]\n")
            else:
                console.print("[dim]Resuming previous session.[/dim]\n")

    review_folder = process_archive(directory, cutoff_year, confidence_threshold, xmp_only, enable_geo, recursive, folder_consensus, dry_run=args.dry_run, skip_dated=args.skip_dated)

    if review_folder and not args.dry_run:
        data = _load_review_json(review_folder)
        if data:
            pending_count = sum(1 for i in data.get("items", []) if i.get("status") == "pending")
            if pending_count:
                if _yn(f"[cyan]{pending_count}[/cyan] photo(s) need review. Run interactive review now?"):
                    run_review_pass(review_folder, xmp_only=xmp_only)
                else:
                    console.print(f"[dim]Run later with:[/dim] [cyan]python metadata-ai.py {shlex.quote(review_folder)} --review[/cyan]")
