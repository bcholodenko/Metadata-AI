import os
import io
import sys
import base64
import re
import time
import threading
import warnings
import json
import shutil
import subprocess
import tempfile
import concurrent.futures
from pathlib import Path
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

# Force line-buffered output so progress prints appear immediately in the terminal.
# reconfigure() is not available on all platforms/builds; fail silently if so.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

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

# Suppress iptcinfo3 logging (it can be very noisy).
logging.getLogger('iptcinfo').setLevel(logging.ERROR)

register_heif_opener()

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_config(path: str = "config.json") -> dict:
    """Load config.json from the script's directory, returning defaults on any error."""
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    config_path  = os.path.join(script_dir, path)
    defaults = {
        "lm_studio": {"url": "http://localhost:1234/v1", "model": "qwen/qwen3.6-27b"},
        "defaults":  {"directory": "./photos", "cutoff_year": 2010,
                      "date_confidence": 7, "geo_confidence": 7},
        "limits":    {"vlm_max_dimension": 2048, "max_image_pixels": 500_000_000,
                      "min_photo_year": 1826, "min_video_year": 1888, "max_year": 2100},
    }
    if not os.path.exists(config_path):
        return defaults
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        # Deep-merge: user values override defaults, missing keys fall back
        for section, section_defaults in defaults.items():
            if section not in data:
                data[section] = section_defaults
            else:
                for key, val in section_defaults.items():
                    data[section].setdefault(key, val)
        return data
    except Exception as e:
        print(f"Warning: could not load config.json ({e}) — using defaults.", flush=True)
        return defaults


_CONFIG = _load_config()

# Raise Pillow's decompression bomb limit to handle large scanned photos.
Image.MAX_IMAGE_PIXELS = _CONFIG["limits"]["max_image_pixels"]

VLM_MAX_DIMENSION = _CONFIG["limits"]["vlm_max_dimension"]

MIN_PHOTO_YEAR = _CONFIG["limits"]["min_photo_year"]
MIN_VIDEO_YEAR = _CONFIG["limits"]["min_video_year"]
MAX_YEAR       = _CONFIG["limits"]["max_year"]

DIRECTORY = _CONFIG["defaults"]["directory"]
MODEL_ID  = _CONFIG["lm_studio"]["model"]
CLIENT    = OpenAI(base_url=_CONFIG["lm_studio"]["url"], api_key="lm-studio")

EXTENSIONS = (
    '.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic',
    '.dng', '.webp', '.cr2', '.cr3', '.nef', '.arw',
    '.raf', '.orf', '.rw2', '.raw',
)
RAW_EXTENSIONS = (
    '.cr2', '.cr3', '.nef', '.arw', '.raf', '.orf', '.rw2', '.raw',
)
VIDEO_EXTENSIONS = (
    '.mp4', '.mov', '.avi', '.m4v', '.mkv', '.mts',
    '.m2ts', '.wmv', '.flv', '.webm',
)

# Fuzzy date patterns — map vague decade/era language to YYYY:MM:DD.
# Word boundaries (\b) prevent matches inside larger tokens like "Photo_2024sample".
FUZZY_DATE_PATTERNS = [
    (r'\bearly\s+(\d{4})s\b',  lambda m: f"{m.group(1)}:01:01 12:00:00"),
    (r'\bmid[- ](\d{4})s\b',   lambda m: f"{str(int(m.group(1))+5)}:01:01 12:00:00"),
    (r'\blate\s+(\d{4})s\b',   lambda m: f"{str(int(m.group(1))+7)}:01:01 12:00:00"),
    (r'\bcirca\s+(\d{4})\b',   lambda m: f"{m.group(1)}:01:01 12:00:00"),
    (r'\bc\.\s*(\d{4})\b',     lambda m: f"{m.group(1)}:01:01 12:00:00"),
    (r'\b(\d{4})s\b',          lambda m: f"{m.group(1)}:01:01 12:00:00"),
]

# Geocoding rate limiting — Nominatim's TOS requires <= 1 request/second.
_LAST_GEOCODE_TIME    = 0.0
_GEOCODE_MIN_INTERVAL = 1.1   # seconds; small buffer over the 1 s minimum
_GEOCODE_CACHE        = {}    # location_text -> (lat, lon) or None

# Folder name patterns that carry no useful context for the VLM.
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

# Time-of-day words filtered out of VLM keyword lists — already captured in
# the dedicated TIME field, no need to duplicate them as tags.
_TIME_WORDS = {
    "morning", "midday", "noon", "afternoon", "evening",
    "night", "dawn", "dusk", "sunrise", "sunset", "golden hour",
}

# Location hedge phrases — indicate the VLM is guessing rather than identifying.
_LOCATION_HEDGE_PHRASES = (
    "no identifiable", "no clear", "cannot identify", "unable to",
    "no location", "no specific", "there are no", "i cannot", "i can't",
    "unsorted", "folder", "unknown", "region", "likely",
)

# ---------------------------------------------------------------------------
# Video prompt templates
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
- LOCATION: specific city or place if clearly identifiable and you have at least 8/10 confidence (visible sign, recognisable landmark, or distinctive architecture); otherwise none — do not guess from general landscape appearance
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
# Platform helpers
# ---------------------------------------------------------------------------


def _quote_path(p: str) -> str:
    """Return a shell-displayable version of *p* for use in hint messages."""
    import platform
    if platform.system() == "Windows":
        return '"' + p.replace('"', '\\"') + '"'
    import shlex
    return shlex.quote(p)


def _strip_shell_escapes(p: str) -> str:
    """Remove bash/zsh backslash-escapes from a pasted path.

    On Windows, backslashes are path separators and must not be removed, so
    the string is returned unchanged.  On POSIX systems the shell typically
    escapes spaces and special characters with a leading backslash; this
    function strips those escape characters so the path resolves correctly.

    Defined at module level (not inside ``__main__``) so it is available to
    callers that import this module rather than running it directly.
    """
    import platform
    if platform.system() == "Windows":
        return p
    return re.sub(r'\\(.)', r'\1', p)


def _resolve_binary(name: str) -> str:
    """Return the full path to an external binary, or *name* if not found."""
    return shutil.which(name) or name


# ---------------------------------------------------------------------------
# Interactive prompt helpers
# ---------------------------------------------------------------------------


def _yn(prompt: str, default_yes: bool = False) -> bool:
    """Display a styled yes/no prompt and return the boolean answer."""
    hint = "[Y/n]" if default_yes else "[y/N]"
    try:
        ans = console.input(f"\n[bold]{prompt}[/bold] {hint} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default_yes
    if not ans:
        return default_yes
    return ans.startswith("y")


def _ask(prompt: str, default: str = "") -> str:
    """Display a styled text prompt and return the user's input."""
    hint = f" [[dim]{default}[/dim]]" if default else ""
    try:
        val = console.input(f"[bold]{prompt}[/bold]{hint} ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return val or default


def _questionary_select(
    prompt: str,
    choices: list[tuple[str, str]],
    default: str,
) -> str:
    """Display an arrow-key selection menu, falling back to a numbered list."""
    try:
        import questionary
        if sys.stdin.isatty():
            q_choices = [
                questionary.Choice(title=f"{k:<22} {desc}", value=k)
                for k, desc in choices
            ]
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
    """Display a Yes/No arrow-key questionary select."""
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
    """Format a duration in seconds as a human-readable ETA string."""
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


# ---------------------------------------------------------------------------
# Pause-on-keypress machinery
#
# A background daemon thread watches stdin for input using select() on POSIX
# or msvcrt.kbhit() on Windows.  When the user types p + Enter the
# _PAUSE_EVENT is set.  The main photo loop checks the event at the top of
# each iteration (between photos, not mid-VLM-call).
#
# Raw-mode tty is deliberately avoided: setraw() intercepts Ctrl-C before
# Python's signal handler can catch it, which makes the app appear frozen.
# Using select() + normal line-mode read leaves signal handling intact.
# ---------------------------------------------------------------------------

_PAUSE_EVENT   = threading.Event()
_STOP_LISTENER = threading.Event()


def _keypress_listener():
    """Background thread: set _PAUSE_EVENT when the user types p (+ Enter)."""
    if not sys.stdin.isatty():
        return
    try:
        import platform
        if platform.system() == "Windows":
            import msvcrt, time as _time
            while not _STOP_LISTENER.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch.lower() == 'p':
                        _PAUSE_EVENT.set()
                else:
                    _time.sleep(0.05)
        else:
            import select
            buf = b""
            while not _STOP_LISTENER.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                try:
                    chunk = os.read(sys.stdin.fileno(), 64)
                except OSError:
                    break
                if not chunk:  # EOF
                    break
                buf += chunk
                # Process any complete lines
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if line.strip().lower() == b'p':
                        _PAUSE_EVENT.set()
    except Exception:
        pass


def start_pause_listener():
    """Start the background keypress listener thread."""
    if not sys.stdin.isatty():
        return
    _STOP_LISTENER.clear()
    t = threading.Thread(target=_keypress_listener, daemon=True, name="pause-listener")
    t.start()


def stop_pause_listener():
    """Signal the keypress listener thread to exit."""
    _STOP_LISTENER.set()


def check_for_pause():
    """If the user pressed p, print a pause message and wait for Enter."""
    if not _PAUSE_EVENT.is_set():
        return
    _PAUSE_EVENT.clear()
    console.print(
        "\n[bold yellow]⏸  Paused[/bold yellow] [dim]— press Enter to continue, "
        "or Ctrl-C to quit (progress is saved).[/dim]"
    )
    try:
        console.input("")
    except KeyboardInterrupt:
        raise   # let Ctrl-C propagate normally
    except EOFError:
        pass
    console.print("[bold green]▶  Resuming…[/bold green]\n")


def _run_vlm_async(fn, *args, poll_interval=0.15, **kwargs):
    """Run a VLM call in a background thread, polling for pause/interrupt.

    Submits fn(*args, **kwargs) to a thread and returns its result.
    While waiting, checks for pause/Ctrl-C every poll_interval seconds
    so the user is not stuck waiting a full VLM round-trip before the
    pause takes effect.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        while True:
            try:
                return future.result(timeout=poll_interval)
            except concurrent.futures.TimeoutError:
                check_for_pause()
            except KeyboardInterrupt:
                future.cancel()
                raise


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def parse_fuzzy_date(text):
    """Extract a normalised EXIF date string from vague or ambiguous text.

    Parameters
    ----------
    text : str or None
        Raw date string from a VLM response or handwritten note.

    Returns
    -------
    date_str : str or None
        EXIF-format date ``"YYYY:MM:DD HH:MM:SS"`` on success, else ``None``.
    raw_text : str or None
        The original fuzzy string (e.g. ``"1970s"``) when a decade pattern
        was matched, so callers can store it for human reference.  ``None``
        when an exact date was parsed.

    Notes
    -----
    Resolution order:

    1. Unambiguous ISO-like formats (``YYYY:MM:DD``, ``YYYY-MM-DD``, …).
    2. Bare four-digit year.
    3. Fuzzy decade patterns from :data:`FUZZY_DATE_PATTERNS`.
    4. VLM normalisation fallback via :func:`normalize_date_with_vlm`.
    """
    if not text:
        return None, None

    m = re.search(r'(\d{4})[:/-](\d{2})[:/-](\d{2})', text)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)} 12:00:00", None

    m = re.search(r'\b(\d{4})\b', text)
    if m:
        return f"{m.group(1)}:01:01 12:00:00", None

    for pattern, formatter in FUZZY_DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return formatter(m), text.strip()

    normalized = normalize_date_with_vlm(text)
    if normalized:
        return normalized, text.strip()

    return None, None


# ---------------------------------------------------------------------------
# IPTC helper
# ---------------------------------------------------------------------------


def get_iptc_metadata(path):
    """Extract existing IPTC keywords from an image file."""
    try:
        info = IPTCInfo(path, force=True)
        return [k.decode('utf-8') for k in info['keywords']] if info['keywords'] else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Image and API helpers
# ---------------------------------------------------------------------------


def normalize_date_with_vlm(raw_text):
    """Ask the VLM to extract a single year from an ambiguous date string."""
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL_ID,
            messages=[{
                "role": "user",
                "content": (
                    f"Extract the most likely single year from this date string: '{raw_text}'. "
                    "Reply with ONLY a 4-digit year, nothing else. "
                    "If it's a range like '1992-93' or '1992-1993', return the start year."
                ),
            }],
            max_tokens=10,
        )
        year_str = response.choices[0].message.content.strip()
        m = re.search(r'\b(\d{4})\b', year_str)
        if m:
            return f"{m.group(1)}:01:01 12:00:00"
    except Exception:
        pass
    return None


def get_jpeg_base64(image_path):
    """Open an image, downscale it, and return a JPEG base64 string."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
            import numpy as np
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            img = Image.fromarray(rgb)
        except ImportError:
            raise RuntimeError(
                "rawpy is required to open raw files.  "
                "Install it with:  pip install rawpy"
            )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = Image.open(image_path)
            img.load()
            img = img.copy()   # detach from the file handle so we can close it

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > VLM_MAX_DIMENSION:
        scale    = VLM_MAX_DIMENSION / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img      = img.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def ask_vlm(image_path, prompt, max_tokens=600):
    """Send an image to the local VLM and return the text response."""
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
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }],
        )
        return response.choices[0].message.content
    except Exception as e:
        console.print(f"      [red]VLM request failed: {e}[/red]")
        return ""


def run_tesseract(image_path):
    """Run Tesseract OCR on an image and return the extracted text."""
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
# Folder name validation
# ---------------------------------------------------------------------------


def is_meaningful_folder_name(name):
    """Return ``True`` if the folder name carries useful context for the VLM."""
    if not name or len(name) < 2:
        return False
    lower = name.lower().strip()
    for pattern in _FOLDER_NOISE_PATTERNS:
        if re.search(pattern, lower):
            return False

    if re.search(r'\b(18|19|20)\d{2}\b', name):
        return True

    if re.search(r'\b\d{1,2}[-/]\d{1,2}([-/]\d{2,4})?\b', name):
        return True

    if re.search(
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|'
        r'january|february|march|april|june|july|august|september|'
        r'october|november|december)[\s.,-]+\d{2,4}\b',
        lower,
    ):
        return True

    words = [w for w in re.findall(r'[A-Za-z]+', name) if len(w) >= 3]
    return len(words) >= 2


# ---------------------------------------------------------------------------
# Metadata writers
# ---------------------------------------------------------------------------


def apply_metadata(
    path,
    date_str,
    tags=None,
    comment=None,
    raw_date=None,
    gps=None,
    xmp_only=False,
    scene=None,
    setting=None,
    flash=None,
):
    """Write metadata to a photo file, choosing the appropriate backend.

    Parameters
    ----------
    path : str
        Absolute path to the image file.
    date_str : str or None
        EXIF-format date (``"YYYY:MM:DD HH:MM:SS"``) or ``None`` to skip the
        date field while still writing other metadata.
    tags : str or None, optional
        Comma-separated keyword string.
    comment : str or None, optional
        Handwritten or VLM-generated comment text.
    raw_date : str or None, optional
        Original fuzzy date string (``"1970s"``) stored in the description.
    gps : tuple or None, optional
        ``(latitude, longitude)`` decimal degrees, or ``None``.
    xmp_only : bool, optional
        When ``True``, write an XMP sidecar only and skip ExifTool merge.
    scene : str or None, optional
        One-sentence VLM scene description.
    setting : str or None, optional
        ``"indoor"`` or ``"outdoor"``.
    flash : str or None, optional
        ``"yes"`` or ``"no"``.

    Notes
    -----
    DNG and raw files always receive XMP sidecars.  JPEG, TIFF, PNG, HEIC,
    and WebP files use ExifTool to merge the sidecar into the file.
    """
    ext = os.path.splitext(path)[1].lower()
    if xmp_only or ext in ('.dng',) + RAW_EXTENSIONS:
        _apply_metadata_xmp(path, date_str, tags, comment, raw_date, gps, scene, setting, flash)
    elif ext in ('.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.webp'):
        _apply_metadata_via_exiftool(path, date_str, tags, comment, raw_date, gps, scene, setting, flash)
    else:
        console.print(f"   [yellow]⚠ Unsupported format for metadata writing: {ext}[/yellow]")


def _apply_metadata_via_exiftool(
    path,
    date_str,
    tags=None,
    comment=None,
    raw_date=None,
    gps=None,
    scene=None,
    setting=None,
    flash=None,
):
    """Write an XMP sidecar and merge it into the image file via ExifTool."""
    xmp_path = os.path.splitext(path)[0] + ".xmp"
    _apply_metadata_xmp(
        path, date_str, tags, comment, raw_date, gps, scene, setting, flash,
        verbose=False,
    )

    if shutil.which("exiftool") is None:
        console.print(
            f"      [yellow]⚠ ExifTool not found — XMP sidecar kept at "
            f"{os.path.basename(xmp_path)}[/yellow]"
        )
        return

    try:
        result = subprocess.run(
            [_resolve_binary("exiftool"), "-overwrite_original", f"-tagsfromfile={xmp_path}", path],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            os.remove(xmp_path)
            console.print(f"   [bold green]✓[/bold green] {os.path.basename(path)}")
        else:
            console.print(
                f"      [yellow]⚠ ExifTool merge failed — XMP sidecar kept. "
                f"Error: {result.stderr.strip()}[/yellow]"
            )
    except Exception as e:
        console.print(f"      [yellow]⚠ ExifTool error — XMP sidecar kept: {e}[/yellow]")


def _xml_escape(s):
    """Escape special XML characters in a string."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _apply_metadata_xmp(
    path,
    date_str,
    tags=None,
    comment=None,
    raw_date=None,
    gps=None,
    scene=None,
    setting=None,
    flash=None,
    verbose=True,
):
    """Write an XMP sidecar file for *path*.

    Parameters
    ----------
    path : str
        Absolute path to the image file.  The sidecar is written alongside it
        with the same stem and a ``.xmp`` extension.
    date_str : str or None
        EXIF-format date or ``None`` to omit the date field.
    tags : str or None, optional
        Comma-separated keywords written as ``dc:subject`` items.
    comment : str or None, optional
        Comment appended to the ``dc:description`` field.
    raw_date : str or None, optional
        Original fuzzy date string stored in the description for reference.
    gps : tuple or None, optional
        ``(latitude, longitude)`` decimal degrees, or ``None``.
    scene : str or None, optional
        One-sentence scene description (primary ``dc:description`` content).
    setting : str or None, optional
        ``"indoor"`` or ``"outdoor"`` — appended to the description.
    flash : str or None, optional
        ``"yes"`` or ``"no"`` — mapped to ``exif:Flash/exif:Fired``.
    verbose : bool, optional
        When ``True`` (default) print a success line after writing.  Set to
        ``False`` when the sidecar is an intermediate file about to be merged
        by ExifTool.

    Notes
    -----
    ``dc:description`` carries the scene sentence, comment, setting, and raw
    date string — there is no standard EXIF tag for indoor/outdoor, so it
    lives here.  Flash maps to EXIF tag 0x9209 when ExifTool merges the file.
    """
    try:
        xmp_path = os.path.splitext(path)[0] + ".xmp"

        keywords_xml = ""
        if tags:
            keywords_xml = "".join(
                f"          <rdf:li>{_xml_escape(kw.strip())}</rdf:li>\n"
                for kw in tags.split(",")
            )

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
            description_xml = (
                f"      <dc:description><rdf:Alt>"
                f"<rdf:li xml:lang='x-default'>{joined}</rdf:li>"
                f"</rdf:Alt></dc:description>\n"
            )

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
            gps_xml = (
                f"      <exif:GPSLatitude>{lat}</exif:GPSLatitude>\n"
                f"      <exif:GPSLongitude>{lon}</exif:GPSLongitude>\n"
            )

        xmp_content = (
            "<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>\n"
            "<x:xmpmeta xmlns:x='adobe:ns:meta/'>\n"
            "  <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>\n"
            "    <rdf:Description xmlns:xmp='http://ns.adobe.com/xap/1.0/'\n"
            "                     xmlns:dc='http://purl.org/dc/elements/1.1/'\n"
            "                     xmlns:exif='http://ns.adobe.com/exif/1.0/'>\n"
        )
        if date_str:
            xmp_content += f"      <exif:DateTimeOriginal>{date_str}</exif:DateTimeOriginal>\n"
        xmp_content += gps_xml + flash_xml + description_xml
        xmp_content += (
            "      <dc:subject>\n"
            "        <rdf:Bag>\n"
            f"{keywords_xml}"
            "        </rdf:Bag>\n"
            "      </dc:subject>\n"
            "    </rdf:Description>\n"
            "  </rdf:RDF>\n"
            "</x:xmpmeta>\n"
            "<?xpacket end='w'?>"
        )

        with open(xmp_path, "w", encoding="utf-8") as f:
            f.write(xmp_content)
        if verbose:
            console.print(f"   [bold green]✓[/bold green] {os.path.basename(xmp_path)}")
    except Exception as e:
        console.print(f"   [red]✗ Metadata Error for {os.path.basename(path)}: {e}[/red]")


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------


def _review_json_path(folder):
    """Return the canonical path to the review JSON file for *folder*."""
    return os.path.join(folder, "review.json")


def _review_html_path(folder):
    """Return the canonical path to the review HTML file for *folder*."""
    return os.path.join(folder, "review.html")


def _load_review_json(folder):
    """Load an existing ``review.json`` for *folder*."""
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
    """Atomically save *data* as ``review.json`` inside *folder*."""
    path = _review_json_path(folder)
    tmp  = path + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        console.print(f"   [yellow]⚠ Could not save review.json: {e}[/yellow]")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _thumbnail_data_uri(image_path, max_dim=240):
    """Return a base64 JPEG data URI thumbnail for *image_path*."""
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
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def write_review_report(folder, review_queue):
    """Write ``review.json`` and ``review.html`` for *review_queue*."""
    if not review_queue:
        return

    existing        = _load_review_json(folder) or {"items": []}
    existing_by_path = {item.get("path"): item for item in existing.get("items", [])}

    enriched = []
    for q in review_queue:
        path   = q.get("path", "")
        prev   = existing_by_path.get(path)
        item_id      = prev.get("id")     if prev else f"item_{abs(hash(path)) % 10**10:010d}"
        status       = prev.get("status") if prev else "pending"
        decided_date = prev.get("decided_date") if prev else None
        thumb        = prev.get("thumb")  if prev and prev.get("thumb") else _thumbnail_data_uri(path)

        enriched.append({
            "id":           item_id,
            "status":       status,
            "decided_date": decided_date,
            "folder":       q.get("folder", ""),
            "file":         q.get("file", ""),
            "path":         path,
            "raw_guess":    q.get("raw_guess", ""),
            "found_date":   q.get("found_date", ""),
            "confidence":   q.get("confidence", 0),
            "comment":      q.get("comment", "") or "",
            "tags":         q.get("tags")    or "",
            "raw_date":     q.get("raw_date") or "",
            "gps":          q.get("gps"),
            "scene":        q.get("scene")   or "",
            "setting":      q.get("setting") or "",
            "flash":        q.get("flash")   or "",
            "thumb":        thumb,
        })

    data = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root":      folder,
        "items":     enriched,
    }
    _save_review_json(folder, data)
    _render_review_html(folder, data)

    pending_count = sum(1 for i in enriched if i["status"] == "pending")
    console.print(f"\n[cyan]📋 Review report:[/cyan] {_review_html_path(folder)}")
    if pending_count != len(enriched):
        console.print(
            f"   [dim]{pending_count} pending, "
            f"{len(enriched) - pending_count} previously decided.[/dim]"
        )


def _render_review_html(folder, data):
    """Regenerate the dark-mode ``review.html`` from *data*."""
    items   = data.get("items", [])
    pending = [i for i in items if i["status"] == "pending"]
    applied = [i for i in items if i["status"] == "applied"]
    skipped = [i for i in items if i["status"] == "skipped"]

    def card(item):
        """Render a single photo card as an HTML snippet."""
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
            comment_html = (
                f'<div class="meta-row"><span class="label">Note:</span> '
                f'<span>{_xml_escape(item["comment"])}</span></div>'
            )

        scene_html = ""
        if item.get("scene"):
            scene_html = f'<div class="meta-row scene">{_xml_escape(item["scene"])}</div>'

        return (
            f'\n    <div class="card {status_class}" data-id="{item["id"]}">\n'
            f'      <div class="thumb">{thumb_html}</div>\n'
            f'      <div class="body">\n'
            f'        <div class="filename">{_xml_escape(item["file"])}</div>\n'
            f'        <div class="folder">{_xml_escape(item.get("folder", ""))}</div>\n'
            f'        {scene_html}\n'
            f'        <div class="meta-row"><span class="label">AI guess:</span> '
            f'<span class="guess">{_xml_escape(item["raw_guess"])}</span></div>\n'
            f'        <div class="meta-row"><span class="label">Confidence:</span> '
            f'<span class="conf">{item["confidence"]}/10</span></div>\n'
            f'        {comment_html}\n'
            f'        <div class="status-pill {status_class}">{_xml_escape(status_label)}</div>\n'
            f'      </div>\n'
            f'    </div>'
        )

    cards_html = "\n".join(card(item) for item in items)
    generated  = data.get("generated", "")

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
    Run <code>python metadata-ai.py {_xml_escape(_quote_path(folder))} --review</code> in your terminal to step through pending items.
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
    """Walk pending review items and prompt the user for a decision on each.

    Parameters
    ----------
    folder : str
        Directory containing ``review.json``.
    xmp_only : bool, optional
        Passed through to :func:`apply_metadata` for accepted items.

    Notes
    -----
    Decisions:

    * ``a`` — accept the AI's date and write metadata immediately.
    * ``e`` — enter a different date and write metadata.
    * ``s`` — skip permanently (no metadata written).
    * ``q`` — quit; all decisions made so far are persisted.

    Each decision is saved before moving to the next photo so a ``Ctrl-C``
    at any point loses at most one decision.  The run can be resumed later
    with ``--review``.
    """
    data = _load_review_json(folder)
    if not data or not data.get("items"):
        console.print("[yellow]No review.json found in this folder — nothing to review.[/yellow]")
        return

    items   = data["items"]
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
        console.print(
            f"[bold][{idx}/{total}][/bold] [cyan]{item['file']}[/cyan]  "
            f"[dim]{item.get('folder', '')}[/dim]"
        )
        if item.get("scene"):
            console.print(f"  [dim]Scene:[/dim]      {item['scene']}")
        console.print(
            f"  [dim]AI guess:[/dim]   [bold]{item['raw_guess']}[/bold]  "
            f"[yellow]({item['confidence']}/10 confidence)[/yellow]"
        )
        if item.get("comment") and item["comment"] not in ("—", ""):
            console.print(f"  [dim]Note:[/dim]       {item['comment']}")

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
                    choice = console.input(
                        "  [bold]Decision[/bold] [[bold cyan]a[/bold cyan]/e/s/q]: "
                    ).strip().lower() or "a"
                except (EOFError, KeyboardInterrupt):
                    choice = "q"
                if choice in ("a", "e", "s", "q"):
                    break
                console.print("  [yellow]Please enter a, e, s, or q.[/yellow]")

        if choice == "q":
            console.print(f"\n[dim]Quitting. {decided_count}/{total} decided this session.[/dim]")
            console.print(
                f"[dim]Resume with:[/dim] [cyan]python metadata-ai.py "
                f"{_quote_path(folder)} --review[/cyan]"
            )
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
                    console.print(
                        f"  [yellow]⚠ Could not parse '{date_to_write}' as a date "
                        f"— use [e] to enter one manually.[/yellow]"
                    )
                    continue
                date_to_write = parsed
        else:
            try:
                user_date = console.input(
                    "  [bold]Date[/bold] [dim](YYYY:MM:DD, YYYY, 'circa 1970s', …)[/dim]: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if not user_date:
                console.print("  [yellow]⚠ Empty input — leaving as pending.[/yellow]")
                continue
            parsed, _ = parse_fuzzy_date(user_date)
            if not parsed:
                console.print(
                    f"  [yellow]⚠ Could not parse '{user_date}' as a date "
                    f"— leaving as pending.[/yellow]"
                )
                continue
            date_to_write = parsed

        try:
            gps = tuple(item["gps"]) if item.get("gps") else None
            apply_metadata(
                item["path"], date_to_write,
                tags    = item.get("tags")    or None,
                comment = item.get("comment") if item.get("comment") not in ("—", "") else None,
                raw_date= item.get("raw_date") or None,
                gps     = gps,
                xmp_only= xmp_only,
                scene   = item.get("scene")   or None,
                setting = item.get("setting") or None,
                flash   = item.get("flash")   or None,
            )
            item["status"]       = "applied"
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
    """Look up GPS coordinates for *location_text* via Nominatim."""
    global _LAST_GEOCODE_TIME
    if not location_text:
        return None
    # Normalise before caching so "Paris", " paris ", and "PARIS" all share
    # the same cache entry AND the same string is sent to Nominatim — fixing
    # the previous mismatch where the cache key was lowercased but the geocode
    # query used the original casing.
    normalised = location_text.strip()
    cache_key  = normalised.lower()
    if cache_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cache_key]
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(
            user_agent="metadata-ai/1.0 (https://github.com/bcholodenko/Metadata-AI)"
        )
        elapsed = time.monotonic() - _LAST_GEOCODE_TIME
        if elapsed < _GEOCODE_MIN_INTERVAL:
            time.sleep(_GEOCODE_MIN_INTERVAL - elapsed)
        location = geolocator.geocode(normalised, timeout=10)
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


def _checkpoint_path(root_folder):
    """Return the path of the progress checkpoint file for *root_folder*."""
    return os.path.join(root_folder, ".metadata-ai-progress")


def _load_checkpoint(root_folder):
    """Load the set of already-processed file paths from the checkpoint."""
    path = _checkpoint_path(root_folder)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def _save_checkpoint(root_folder, completed_path):
    """Append *completed_path* to the checkpoint file for *root_folder*."""
    with open(_checkpoint_path(root_folder), 'a', encoding='utf-8') as f:
        f.write(completed_path + '\n')


def _clear_checkpoint(root_folder):
    """Delete the checkpoint file for *root_folder* if it exists."""
    path = _checkpoint_path(root_folder)
    if os.path.exists(path):
        os.remove(path)


def _has_existing_date(path):
    """Return ``True`` if *path* already has a ``DateTimeOriginal`` tag."""
    if not shutil.which("exiftool"):
        return False
    try:
        result = subprocess.run(
            [_resolve_binary("exiftool"), "-DateTimeOriginal", "-s3", path],
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# VLM field helpers
# ---------------------------------------------------------------------------


def _clean_vlm_field(s):
    """Strip markdown formatting characters from a VLM field value."""
    return re.sub(r'[*_`#]', '', s).strip() if s else s


def _parse_time_of_day(text):
    """Convert a natural-language time string to a 24-hour integer hour."""
    if not text:
        return None
    text = text.lower().strip()

    m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', text)
    if m:
        hour = int(m.group(1))
        ampm = m.group(3)
        # Reject hours outside the valid 12-hour clock range (1–12).
        # Silently clamping an impossible value like "13pm" to 23 would write
        # wrong EXIF data; returning None is safer and keeps the field unset.
        if not (1 <= hour <= 12):
            return None
        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        return hour  # guaranteed 0–23; no clamp needed after range check above

    m = re.search(r'\b(\d{1,2}):(\d{2})\b', text)
    if m:
        return min(int(m.group(1)), 23)

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
    return None


# ---------------------------------------------------------------------------
# PhotoAnalysis dataclass
# ---------------------------------------------------------------------------


@dataclass
class PhotoAnalysis:
    """Collects everything extracted about one photo before the write step.

    Attributes
    ----------
    found_date : str or None
        EXIF-format date ``"YYYY:MM:DD HH:MM:SS"``, or ``None``.
    raw_date_text : str or None
        Original fuzzy string (e.g. ``"1970s"``) kept for sidecar reference.
    confidence : int
        Date confidence on a 1-10 scale.  Defaults to 10 (certainty level
        of back-of-photo or IPTC dates); reduced to the VLM's self-reported
        score when step 3 is the source.
    comment : str or None
        Handwritten or printed text found on the back of the photo.
    scene : str or None
        One-sentence VLM scene description.
    setting : str or None
        ``"indoor"`` or ``"outdoor"``.
    flash : str or None
        ``"yes"`` or ``"no"``.
    tags : str or None
        Comma-separated keyword string.
    gps : tuple or None
        ``(latitude, longitude)`` decimal degrees, or ``None``.
    time_hour : int or None
        Hour of day (0-23) parsed from the VLM's time estimate, or ``None``.

    Notes
    -----
    Each field is independently optional.  A photo with a clear date from
    the back but no VLM description still yields a valid ``PhotoAnalysis``
    with most fields as ``None``.
    """

    found_date:    Optional[str]   = None
    raw_date_text: Optional[str]   = None
    confidence:    int             = 10
    comment:       Optional[str]   = None
    scene:         Optional[str]   = None
    setting:       Optional[str]   = None
    flash:         Optional[str]   = None
    tags:          Optional[str]   = None
    gps:           Optional[tuple] = None
    time_hour:     Optional[int]   = None


# ---------------------------------------------------------------------------
# Per-photo analysis helpers
# ---------------------------------------------------------------------------


def _try_back_of_photo(folder, files, i, processed_files):
    """Phase 1: Check whether ``files[i+1]`` is the back side of ``files[i]``.

    Parameters
    ----------
    folder : str
        Absolute path to the folder containing *files*.
    files : list of str
        Sorted filename list for the current folder.
    i : int
        Index of the current (front) photo in *files*.
    processed_files : set of str
        Mutable set of filenames already consumed.  The next file is added
        here when confirmed as a back.

    Returns
    -------
    found_date : str or None
        EXIF-format date extracted from the back, or ``None``.
    raw_date_text : str or None
        Original fuzzy date string, or ``None``.
    comment : str or None
        Handwritten comment text found on the back, or ``None``.

    Notes
    -----
    A single VLM call simultaneously detects the back, extracts any written
    date, and extracts any comment text.  Tesseract OCR is used as a fallback
    when the VLM returns no date.
    """
    console.print(f"   [dim]1) Back-of-photo check…[/dim]")
    if i + 1 >= len(files):
        console.print(f"      [dim]No next image to check.[/dim]")
        return None, None, None

    next_file = files[i + 1]
    next_path = os.path.join(folder, next_file)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _test = Image.open(next_path)
            _test.load()
            _test.close()
    except Exception:
        console.print(f"      [yellow]⚠ Could not open {next_file} — skipping back check.[/yellow]")
        return None, None, None

    back_prompt = (
        "Look very carefully at this image. "
        "First determine: is this the BACK (reverse side) of a physical printed photograph? "
        "The back shows blank paper, handwriting, stamps, photo lab printing, or a plain surface — no photographic scene. "
        "Reply in this exact format:\n"
        "IS_BACK: <yes or no>\n"
        "DATE: <any date written on it in YYYY:MM:DD format, or 'circa 1950s', or 'none'>\n"
        "COMMENT: <any other handwritten or printed text excluding dates, translated to English, or 'none'>"
    )
    back_resp    = _run_vlm_async(ask_vlm, next_path, back_prompt)
    is_back_line = re.search(r'IS_BACK:\s*([^\n]+)', back_resp or "")
    if not (is_back_line and is_back_line.group(1).strip().lower().startswith("yes")):
        console.print(f"      [dim]No back detected.[/dim]")
        return None, None, None

    console.print(f"      [green]Back confirmed:[/green] {next_file}")
    processed_files.add(next_file)

    date_match    = re.search(r'DATE:\s*([^\n]+)',    back_resp)
    comment_match = re.search(r'COMMENT:\s*([^\n]+)', back_resp)
    raw_date_str  = date_match.group(1).strip()    if date_match    else "none"
    raw_comment   = comment_match.group(1).strip() if comment_match else "none"

    ocr_context = run_tesseract(next_path)
    if ocr_context and raw_date_str.lower() == "none":
        console.print(f"      [dim]Tesseract OCR found text — re-checking for date…[/dim]")
        ocr_prompt   = (
            f"This is the back of a photo. OCR extracted: {ocr_context}\n"
            "Extract any date from this text in YYYY:MM:DD format, or 'circa 1950s', etc. "
            "If no date, return 'none'."
        )
        raw_date_str = _run_vlm_async(ask_vlm, next_path, ocr_prompt).strip()

    found_date, raw_date_text = parse_fuzzy_date(raw_date_str)
    if found_date:
        fuzzy_note = f" [dim](fuzzy: {raw_date_text})[/dim]" if raw_date_text else ""
        console.print(f"      [dim]Date from back:[/dim] [bold]{found_date}[/bold]{fuzzy_note}")
    else:
        console.print(f"      [dim]No date on back — falling through to VLM guess.[/dim]")

    comment_out = None
    if raw_comment.lower() != "none" and raw_comment:
        comment_out = raw_comment
        console.print(f"      [dim]Comment:[/dim] {comment_out}")
    else:
        console.print(f"      [dim]No comment on back.[/dim]")

    return found_date, raw_date_text, comment_out


def _try_iptc_date(path):
    """Phase 2: Scan existing IPTC keywords for a parseable date."""
    console.print(f"   [dim]2) IPTC keyword check…[/dim]")
    iptc_keywords = get_iptc_metadata(path)
    if not iptc_keywords:
        console.print(f"      [dim]No IPTC keywords found.[/dim]")
        return None, None

    for keyword in iptc_keywords:
        if not re.search(r'\d', keyword):
            continue
        found_date, raw_date_text = parse_fuzzy_date(keyword)
        if not found_date:
            continue
        try:
            yr = int(found_date.split(':')[0])
        except (ValueError, IndexError):
            continue
        if not (MIN_PHOTO_YEAR <= yr <= MAX_YEAR):
            continue
        fuzzy_note = f" [dim](fuzzy: {raw_date_text})[/dim]" if raw_date_text else ""
        console.print(
            f"      [dim]Date from IPTC '[/dim][bold]{keyword}[/bold][dim]':[/dim] "
            f"[bold]{found_date}[/bold]{fuzzy_note}"
        )
        return found_date, raw_date_text

    console.print(f"      [dim]No date found in IPTC keywords.[/dim]")
    return None, None


def _vlm_estimate_date(image_path, folder_hint, cutoff_year, confidence_threshold):
    """Phase 3a: Ask the VLM for a focused date estimate.

    Parameters
    ----------
    image_path : str
        Absolute path to the photo.
    folder_hint : str
        Context sentence about the folder name, or an empty string.
    cutoff_year : int
        Upper year bound communicated to the VLM in the prompt.
    confidence_threshold : int
        Threshold used only to colour-code the confidence display.

    Returns
    -------
    found_date : str or None
        Validated EXIF-format date, or ``None``.
    raw_date_text : str or None
        Original fuzzy string, or ``None``.
    confidence : int
        VLM self-reported confidence (1-10); 0 when no response or no date.

    Notes
    -----
    A separate date-only call (rather than combining date + description into
    one prompt) measurably improves confidence calibration.
    """
    date_prompt = (
        "Analyze the fashion, hairstyles, technology, and setting in this photo to estimate when it was taken. "
        f"{folder_hint}"
        "Estimate the date as specifically as possible — could be YYYY:MM:DD, YYYY:MM, YYYY, "
        "a decade like '1970s', or 'circa 1965'. "
        f"The date must be before {cutoff_year}. "
        "Also provide a confidence score from 1-10 for your date estimate.\n"
        "Reply in EXACTLY this format with no other text:\n"
        "DATE: <your estimate>\n"
        "CONFIDENCE: <score 1-10>"
    )
    console.print(f"      [dim]Dating…[/dim]")
    date_resp = _run_vlm_async(ask_vlm, image_path, date_prompt, max_tokens=200) or ""
    if not date_resp:
        console.print(f"      [yellow]⚠ VLM returned no response for date estimate.[/yellow]")
        return None, None, 0

    date_match = re.search(r'DATE:\s*([^\n]+)',    date_resp)
    conf_match = re.search(r'CONFIDENCE:\s*(\d+)', date_resp)
    if not date_match:
        return None, None, 0

    raw_guess  = date_match.group(1).strip()
    confidence = int(conf_match.group(1)) if conf_match else 5
    found_date, raw_date_text = parse_fuzzy_date(raw_guess)

    if not found_date:
        console.print(
            f"      [dim]VLM could not determine a date. "
            f"Raw: '[/dim]{raw_guess}[dim]' ({confidence}/10)[/dim]"
        )
        return None, None, confidence

    parts = found_date.split(':')
    try:
        year_val  = int(parts[0])
        month_val = int(parts[1])
        day_val   = int(parts[2].split()[0])
        if not (MIN_PHOTO_YEAR <= year_val <= MAX_YEAR and 1 <= month_val <= 12 and 1 <= day_val <= 31):
            console.print(f"      [yellow]⚠ Invalid date from VLM ('{raw_guess}') — discarding.[/yellow]")
            return None, None, confidence
    except (IndexError, ValueError):
        console.print(f"      [yellow]⚠ Invalid date format ('{raw_guess}') — discarding.[/yellow]")
        return None, None, confidence

    fuzzy_note = f" [dim]— fuzzy: {raw_date_text}[/dim]" if raw_date_text else ""
    conf_color = "green" if confidence >= confidence_threshold else "yellow" if confidence >= 5 else "red"
    console.print(
        f"      [dim]Date:[/dim]       [bold]{found_date}[/bold]  "
        f"[[{conf_color}]{confidence}/10[/{conf_color}]]{fuzzy_note}"
    )
    return found_date, raw_date_text, confidence


def _vlm_describe_photo(image_path, folder_hint_loc, enable_geo):
    """Phase 3b: Ask the VLM for time, scene, setting, flash, location, keywords.

    Parameters
    ----------
    image_path : str
        Absolute path to the photo.
    folder_hint_loc : str
        Context sentence about the folder name for location inference, or
        an empty string.
    enable_geo : bool
        When ``True``, include ``LOCATION`` and ``LOCATION_CONFIDENCE``
        fields in the prompt.

    Returns
    -------
    dict
        Keys: ``time_hour``, ``raw_time``, ``scene``, ``setting``,
        ``flash``, ``tags``, ``geo_inline``, ``geo_confidence``.

    Notes
    -----
    ``geo_inline`` and ``geo_confidence`` are populated only when
    *enable_geo* is ``True``; otherwise they are ``None`` and ``0``.
    """
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

    console.print(f"      [dim]Describing…[/dim]")
    resp = _run_vlm_async(ask_vlm, image_path, desc_prompt) or ""
    if not resp:
        console.print(f"      [yellow]⚠ VLM returned no response — skipping description.[/yellow]")

    time_match     = re.search(r'TIME:\s*([^\n]+)',             resp)
    scene_match    = re.search(r'SCENE:\s*([^\n]+)',            resp)
    setting_match  = re.search(r'SETTING:\s*([^\n]+)',          resp)
    flash_match    = re.search(r'FLASH:\s*([^\n]+)',            resp)
    geo_match      = re.search(r'LOCATION:\s*([^\n]+)',         resp) if enable_geo else None
    geo_conf_match = re.search(r'LOCATION_CONFIDENCE:\s*(\d+)', resp) if enable_geo else None
    keywords_match = re.search(r'KEYWORDS:\s*([^\n]+)',         resp)

    raw_time_str = _clean_vlm_field(time_match.group(1)) if time_match else None
    time_hour    = _parse_time_of_day(raw_time_str)
    raw_time     = raw_time_str if time_hour is not None else None

    vlm_scene   = _clean_vlm_field(scene_match.group(1)) if scene_match else None
    raw_setting = _clean_vlm_field(setting_match.group(1)) if setting_match else None
    vlm_setting = raw_setting if raw_setting and len(raw_setting) < 30 else None

    raw_flash = _clean_vlm_field(flash_match.group(1)).lower() if flash_match else None
    vlm_flash = None
    if raw_flash:
        if   raw_flash.startswith('yes'): vlm_flash = 'yes'
        elif raw_flash.startswith('no'):  vlm_flash = 'no'

    raw_keywords = _clean_vlm_field(keywords_match.group(1)) if keywords_match else None
    tags_out = None
    if raw_keywords:
        filtered = [
            k.strip().lower() for k in raw_keywords.split(',')
            if k.strip().lower() not in _TIME_WORDS
        ]
        tags_out = ', '.join(filtered) if filtered else None

    geo_inline     = _clean_vlm_field(geo_match.group(1)) if geo_match else None
    geo_confidence = int(geo_conf_match.group(1)) if geo_conf_match else 0

    if raw_time:    console.print(f"      [dim]Time:[/dim]       {raw_time}")
    if vlm_scene:   console.print(f"      [dim]Scene:[/dim]      {vlm_scene}")
    if vlm_setting: console.print(f"      [dim]Setting:[/dim]    {vlm_setting}")
    if vlm_flash:   console.print(f"      [dim]Flash:[/dim]      {vlm_flash}")
    if tags_out:    console.print(f"      [dim]Keywords:[/dim]   [dim]{tags_out}[/dim]")

    return {
        'time_hour':      time_hour,
        'raw_time':       raw_time,
        'scene':          vlm_scene,
        'setting':        vlm_setting,
        'flash':          vlm_flash,
        'tags':           tags_out,
        'geo_inline':     geo_inline,
        'geo_confidence': geo_confidence,
    }


def _resolve_location(geo_resp_inline, geo_confidence, folder_name, threshold):
    """Phase 4: Validate the VLM's location guess and geocode it.

    Parameters
    ----------
    geo_resp_inline : str or None
        Raw location string from the VLM description call.
    geo_confidence : int
        VLM self-reported location confidence (1-10).
    folder_name : str
        Bare folder name used to reject responses that merely echo the folder.
    threshold : int
        Minimum confidence required to attempt geocoding.

    Returns
    -------
    tuple or None
        ``(latitude, longitude)`` on success; ``None`` when either gate
        fails or Nominatim does not recognise the place.

    Notes
    -----
    Two gates are applied in order:

    1. **Basic validity** — response must be non-empty, not a hedge phrase,
       and not simply the folder name repeated.
    2. **Confidence gate** — ``geo_confidence`` must meet *threshold*.
       Responses below threshold are logged and discarded.

    When Nominatim fails on the full string, a simplified version (text
    before the first comma) is tried once before giving up.
    """
    console.print(f"   [dim]4) Location check…[/dim]")
    geo_resp = geo_resp_inline or ""
    geo_resp = re.sub(r'\s*\(.*?\)', '', geo_resp).strip()
    loc_conf = geo_confidence

    is_valid = (
        geo_resp.lower() not in ("", "none")
        and len(geo_resp) < 80
        and geo_resp.lower() != folder_name.lower()
        and not any(phrase in geo_resp.lower() for phrase in _LOCATION_HEDGE_PHRASES)
    )

    if is_valid and loc_conf < threshold:
        console.print(
            f"      [dim]Location skipped:[/dim] [yellow]{geo_resp}[/yellow] "
            f"[dim](confidence {loc_conf}/10 < {threshold} — landscape guess)[/dim]"
        )
        return None

    if not is_valid:
        if geo_resp and geo_resp.lower() not in ("", "none"):
            console.print(f"      [dim]Location rejected:[/dim] [dim]{geo_resp}[/dim]")
        else:
            console.print(f"      [dim]No location identified.[/dim]")
        return None

    console.print(f"      [dim]Location:[/dim]   {geo_resp} [dim]({loc_conf}/10)[/dim]")
    coords = geolocate(geo_resp)
    if coords:
        console.print(f"      [dim]GPS:[/dim]        {coords[0]:.4f}, {coords[1]:.4f}")
        return coords

    simplified = geo_resp.split(',')[0].strip()
    if simplified and simplified.lower() != geo_resp.lower():
        console.print(f"      [dim]Retrying with:[/dim] '{simplified}'")
        coords = geolocate(simplified)
        if coords:
            console.print(f"      [dim]GPS:[/dim]        {coords[0]:.4f}, {coords[1]:.4f}")
            return coords

    console.print(f"      [dim]Could not resolve GPS — skipping.[/dim]")
    return None


def _apply_consensus_year(results, confidence_threshold):
    """Phase 6: Rewrite low-confidence dates to the folder's modal year.

    Parameters
    ----------
    results : list of dict
        Deferred write records collected during the folder pass.  Mutated
        in place when a majority consensus is found.
    confidence_threshold : int
        Minimum confidence for a result to contribute to the vote.

    Returns
    -------
    int or None
        The consensus year when a strict majority exists; ``None`` otherwise.

    Notes
    -----
    A consensus is only declared when the modal year holds a **strict
    majority** (more than 50 %) of the high-confidence votes.  A weak
    plurality (e.g. 2/8 distinct years) is treated as no consensus — the
    low-confidence photos go to the review queue rather than being silently
    force-aligned to a noisy mode.  This prevents incorrect mass-dating of
    rolls where the VLM is inconsistent across visually similar scenes.

    Implementation note: the majority test is ``votes * 2 > total`` (integer
    arithmetic).  This is equivalent to ``votes > total / 2`` but avoids
    floating-point division.  A tie (``votes * 2 == total``) is intentionally
    **not** treated as a majority — do not change ``>`` to ``>=``.
    """
    high_conf_years = [
        int(r['found_date'][:4])
        for r in results
        if r['confidence'] >= confidence_threshold
    ]
    if not high_conf_years:
        console.print(f"\n   [yellow]⚠ No high-confidence results to derive consensus year.[/yellow]")
        return None

    modal_year, votes = Counter(high_conf_years).most_common(1)[0]
    total = len(high_conf_years)

    if votes * 2 <= total:
        console.print(
            f"\n   [yellow]⚠ No clear consensus year[/yellow] "
            f"[dim](top year {modal_year} has only {votes}/{total} votes — needs a majority). "
            f"Low-confidence photos will go to the review queue.[/dim]"
        )
        return None

    console.print(
        f"\n   [cyan]🗳 Consensus year: {modal_year}[/cyan] "
        f"[dim]({votes}/{total} high-confidence result(s))[/dim]"
    )
    for r in results:
        if r['confidence'] < confidence_threshold:
            r['found_date'] = f"{modal_year}:{r['found_date'][5:]}"
            console.print(f"      [dim]📅 {r['file']}: overridden to {r['found_date']} via consensus[/dim]")

    return modal_year


# ---------------------------------------------------------------------------
# Core folder processor
# ---------------------------------------------------------------------------


def _process_folder(
    folder,
    files,
    cutoff_year,
    confidence_threshold,
    xmp_only,
    enable_geo,
    global_offset=0,
    global_total=None,
    folder_consensus=False,
    root_folder=None,
    dry_run=False,
    skip_dated=False,
    review_queue_accumulator=None,
    shared_durations=None,
    batch_totals=None,
    geo_confidence_threshold=7,
    run_log=None,
    work_offset=0,
    work_total=None,
):
    """Process one folder of photos through the full five-step pipeline.

    Parameters
    ----------
    folder : str
        Absolute path to the folder being processed.
    files : list of str
        Natural-sorted list of filenames inside *folder*.
    cutoff_year : int
        Photos dated from this year or later are skipped.
    confidence_threshold : int
        Minimum VLM date confidence required to write without review (1-10).
    xmp_only : bool
        Write XMP sidecars only; skip ExifTool merging.
    enable_geo : bool
        Enable location identification and Nominatim geocoding.
    global_offset : int, optional
        Number of photos already processed in this run (for ETA display).
    global_total : int or None, optional
        Total photo count across the entire run (for ETA display).
    folder_consensus : bool, optional
        Defer writes and apply the folder's modal year to low-confidence
        dates after all photos are analysed.
    root_folder : str or None, optional
        Root directory used for checkpoint storage in recursive mode.
    dry_run : bool, optional
        Log actions without modifying any files.
    skip_dated : bool, optional
        Skip photos that already carry a ``DateTimeOriginal`` tag.
    review_queue_accumulator : list or None, optional
        Shared list to which low-confidence items are appended in recursive
        mode.  When ``None`` a folder-local ``review.json`` is written.
    shared_durations : deque or None, optional
        Shared rolling-window deque for cross-folder ETA computation in
        recursive mode.  A local deque is created when ``None``.
    batch_totals : dict or None, optional
        Mutable dict with keys ``'scanned'``, ``'date_written'``,
        ``'tagged_no_date'``, ``'nothing_written'``, ``'reviewed'``,
        ``'cutoff_skip'``, ``'backs'``.  Accumulated in place when provided.
    geo_confidence_threshold : int, optional
        Minimum VLM location confidence required before geocoding.
        Default 7.

    Returns
    -------
    dict
        Counters dict with the same keys as *batch_totals*.

    Notes
    -----
    The five steps per photo are:

    1. Back-of-photo detection (next file in filename order).
    2. IPTC keyword date extraction.
    3. VLM date estimation (3a) + description (3b).
    4. Location resolution via Nominatim (only when *enable_geo* is ``True``).
    5. Metadata write decision.
    """
    processed_files = set()
    review_queue    = []
    results         = []
    completed_paths = _load_checkpoint(root_folder or folder) if not dry_run else set()

    # Silent-skip counters — flushed as one summary line when a real photo appears
    checkpoint_skip_count = 0
    skip_dated_count      = 0

    if dry_run and os.path.exists(_checkpoint_path(root_folder or folder)):
        console.print("   [dim]ℹ Dry-run: ignoring existing checkpoint, re-analyzing all files.[/dim]")

    tagged_no_date_count  = 0
    nothing_written_count = 0
    cutoff_skip_count     = 0

    folder_name        = os.path.basename(folder)
    folder_name_useful = is_meaningful_folder_name(folder_name)
    hint               = "" if folder_name_useful else "  [dim](low-signal name — not passed to VLM)[/dim]"
    console.print(f"   [bold]📁 {folder_name}[/bold]{hint}")
    console.print(f"   [dim]Archiving {len(files)} photo(s)…[/dim]\n")

    if run_log is not None:
        run_log['folders'].append({'name': folder_name, 'path': folder})

    folder_hint_date = (
        f"The folder containing this photo is named '{folder_name}' — "
        "treat this as high-confidence information for the date. "
        if folder_name_useful else ""
    )
    folder_hint_loc = (
        f"The folder containing this photo is named '{folder_name}' — "
        "treat this as high-confidence information for the location. "
        if folder_name_useful else ""
    )

    WINDOW_SIZE = 5
    MIN_SAMPLES = 3
    _durations: deque = (
        shared_durations if shared_durations is not None
        else deque(maxlen=WINDOW_SIZE)
    )
    _eta_str = ""

    # _work_pos and _work_total drive the [X/N] counter shown on each photo.
    # work_offset is the count of real (non-skipped) photos processed in prior
    # folders this run; work_total is the total remaining across the whole run.
    # Both are computed once in process_archive and passed in so the counter
    # never resets between folders.
    already_done_in_run = len(completed_paths)
    _work_total = work_total if work_total is not None else max(
        (global_total - already_done_in_run) if global_total is not None
        else sum(1 for f in files if os.path.join(folder, f) not in completed_paths),
        1,
    )
    _work_pos = work_offset  # carries across folder boundaries

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

            check_for_pause()

            current_path = os.path.join(folder, current_file)
            _photo_start = time.monotonic()
            progress.update(task, description=f"[bold cyan]{current_file}[/bold cyan]")

            if not os.path.isfile(current_path):
                console.print(
                    f"  [yellow]⚠ File no longer exists — skipping: {current_file}[/yellow]"
                )
                nothing_written_count += 1
                progress.advance(task)
                continue

            if current_path in completed_paths:
                checkpoint_skip_count += 1
                progress.advance(task)
                continue

            if skip_dated and _has_existing_date(current_path):
                skip_dated_count += 1
                progress.advance(task)
                continue

            # Flush any accumulated silent skips before printing this photo's header
            if checkpoint_skip_count or skip_dated_count:
                parts = []
                if checkpoint_skip_count:
                    parts.append(f"{checkpoint_skip_count} already processed")
                if skip_dated_count:
                    parts.append(f"{skip_dated_count} already dated")
                console.print(f"  [dim]⏩ Skipped {' + '.join(parts)}.[/dim]")
                checkpoint_skip_count = 0
                skip_dated_count      = 0

            _work_pos += 1
            console.print(f"\n  [bold][{_work_pos}/{_work_total}][/bold] [cyan]{current_file}[/cyan]")
            analysis = PhotoAnalysis()

            # Step 1: back-of-photo check
            found_date, raw_date_text, comment = _try_back_of_photo(
                folder, files, i, processed_files,
            )
            if found_date:
                analysis.found_date    = found_date
                analysis.raw_date_text = raw_date_text
            if comment:
                analysis.comment = comment

            # Step 2: IPTC keyword date
            if not analysis.found_date:
                found_date, raw_date_text = _try_iptc_date(current_path)
                if found_date:
                    analysis.found_date    = found_date
                    analysis.raw_date_text = raw_date_text

            # Step 3: VLM analysis
            console.print(f"   [dim]3) Analyzing image…[/dim]")
            if not analysis.found_date:
                found_date, raw_date_text, confidence = _vlm_estimate_date(
                    current_path, folder_hint_date, cutoff_year, confidence_threshold,
                )
                if found_date:
                    analysis.found_date    = found_date
                    analysis.raw_date_text = raw_date_text
                    analysis.confidence    = confidence

            desc               = _vlm_describe_photo(current_path, folder_hint_loc, enable_geo)
            analysis.time_hour = desc['time_hour']
            analysis.scene     = desc['scene']
            analysis.setting   = desc['setting']
            analysis.flash     = desc['flash']
            analysis.tags      = desc['tags']

            if analysis.found_date:
                applied_hour        = analysis.time_hour if analysis.time_hour is not None else 12
                analysis.found_date = analysis.found_date[:11] + f"{applied_hour:02d}:00:00"
                time_note           = f" [dim](~{desc['raw_time']})[/dim]" if desc['raw_time'] else ""
                console.print(
                    f"      [dim]Timestamp:[/dim]  [bold]{analysis.found_date}[/bold]{time_note}"
                )

            # Step 4: location resolution
            if enable_geo:
                analysis.gps = _resolve_location(
                    desc['geo_inline'], desc['geo_confidence'],
                    folder_name, geo_confidence_threshold,
                )

            # Step 5: write decision
            console.print(f"   [dim]5) Writing metadata…[/dim]")
            if analysis.found_date:
                try:
                    year = int(analysis.found_date[:4])
                except (ValueError, IndexError):
                    console.print(f"      [red]✗ Malformed date: {analysis.found_date}[/red]")
                    year = None

                if year is None:
                    pass
                elif year >= cutoff_year:
                    console.print(
                        f"      [dim]⏭ Skipping: date {year} is {cutoff_year} or later.[/dim]"
                    )
                    cutoff_skip_count += 1
                elif folder_consensus:
                    results.append({
                        "file":       current_file,
                        "path":       current_path,
                        "found_date": analysis.found_date,
                        "confidence": analysis.confidence,
                        "tags":       analysis.tags,
                        "comment":    analysis.comment,
                        "raw_date":   analysis.raw_date_text,
                        "gps":        analysis.gps,
                        "scene":      analysis.scene,
                        "setting":    analysis.setting,
                        "flash":      analysis.flash,
                    })
                    console.print(f"      [dim]Queued for consensus write.[/dim]")
                elif analysis.confidence >= confidence_threshold:
                    if dry_run:
                        console.print(
                            f"      [dim][DRY RUN] Would write: {analysis.found_date} "
                            f"| tags: {analysis.tags}[/dim]"
                        )
                    else:
                        apply_metadata(
                            current_path, analysis.found_date,
                            tags    = analysis.tags,
                            comment = analysis.comment,
                            raw_date= analysis.raw_date_text,
                            gps     = analysis.gps,
                            xmp_only= xmp_only,
                            scene   = analysis.scene,
                            setting = analysis.setting,
                            flash   = analysis.flash,
                        )
                        _save_checkpoint(root_folder or folder, current_path)
                else:
                    console.print(
                        f"      [yellow]⚠ Low confidence ({analysis.confidence}/10) "
                        f"— added to review queue.[/yellow]"
                    )
                    review_queue.append({
                        "folder":     folder_name,
                        "file":       current_file,
                        "path":       current_path,
                        "raw_guess":  analysis.raw_date_text or analysis.found_date,
                        "found_date": analysis.found_date,
                        "confidence": analysis.confidence,
                        "comment":    analysis.comment or "—",
                        "tags":       analysis.tags,
                        "raw_date":   analysis.raw_date_text,
                        "gps":        list(analysis.gps) if analysis.gps else None,
                        "scene":      analysis.scene,
                        "setting":    analysis.setting,
                        "flash":      analysis.flash,
                    })
            else:
                has_something = any([
                    analysis.tags, analysis.scene, analysis.setting,
                    analysis.flash, analysis.comment, analysis.gps,
                ])
                if has_something:
                    console.print(
                        f"      [yellow]⚠ No date found — writing description/keywords only.[/yellow]"
                    )
                    if dry_run:
                        console.print(
                            f"      [dim][DRY RUN] Would write: tags: {analysis.tags} "
                            f"| scene: {analysis.scene}[/dim]"
                        )
                    else:
                        apply_metadata(
                            current_path, None,
                            tags    = analysis.tags,
                            comment = analysis.comment,
                            raw_date= analysis.raw_date_text,
                            gps     = analysis.gps,
                            xmp_only= xmp_only,
                            scene   = analysis.scene,
                            setting = analysis.setting,
                            flash   = analysis.flash,
                        )
                        _save_checkpoint(root_folder or folder, current_path)
                    tagged_no_date_count += 1
                else:
                    console.print(f"      [red]✗ No date or description found — skipping.[/red]")
                    nothing_written_count += 1

            # Rolling ETA update
            _durations.append(time.monotonic() - _photo_start)
            remaining = max(0, _work_total - _work_pos)
            if len(_durations) >= MIN_SAMPLES and remaining > 0:
                avg_sec  = sum(_durations) / len(_durations)
                _eta_str = f"[dim]{_format_eta(avg_sec * remaining)} remaining[/dim]"
            elif remaining == 0:
                _eta_str = ""
            progress.update(task, eta=_eta_str)
            progress.advance(task)

            # Record to run log
            if run_log is not None:
                geo_label = desc.get('geo_inline') or None
                if geo_label and geo_label.lower() in ('none', ''):
                    geo_label = None
                run_log['photos'].append({
                    'file':       current_file,
                    'folder':     folder_name,
                    'date':       analysis.found_date,
                    'decade':     (int(analysis.found_date[:4]) // 10 * 10) if analysis.found_date else None,
                    'confidence': analysis.confidence,
                    'tags':       analysis.tags,
                    'scene':      analysis.scene,
                    'setting':    analysis.setting,
                    'flash':      analysis.flash,
                    'gps':        list(analysis.gps) if analysis.gps else None,
                    'geo_label':  geo_label,
                })

    # Flush any skips that accumulated in the final stretch of the folder
    if checkpoint_skip_count or skip_dated_count:
        parts = []
        if checkpoint_skip_count:
            parts.append(f"{checkpoint_skip_count} already processed")
        if skip_dated_count:
            parts.append(f"{skip_dated_count} already dated")
        console.print(f"  [dim]⏩ Skipped {' + '.join(parts)}.[/dim]")

    # Folder consensus pass
    if folder_consensus and results:
        consensus_year = _apply_consensus_year(results, confidence_threshold)
        for r in results:
            # Write if: high-confidence on its own, OR consensus year was applied
            # (in which case _apply_consensus_year already rewrote r['found_date']).
            # Do NOT write low-confidence results when no consensus was reached.
            should_write = (
                r['confidence'] >= confidence_threshold
                or (consensus_year is not None and r['confidence'] < confidence_threshold)
            )
            if should_write:
                if dry_run:
                    console.print(
                        f"      [dim][DRY RUN] Would write: {r['found_date']} "
                        f"| tags: {r['tags']}[/dim]"
                    )
                else:
                    apply_metadata(
                        r['path'], r['found_date'],
                        tags    = r['tags'],
                        comment = r['comment'],
                        raw_date= r['raw_date'],
                        gps     = r['gps'],
                        xmp_only= xmp_only,
                        scene   = r['scene'],
                        setting = r['setting'],
                        flash   = r['flash'],
                    )
                    _save_checkpoint(root_folder or folder, r['path'])
            else:
                review_queue.append({
                    "folder":     folder_name,
                    "file":       r['file'],
                    "path":       r['path'],
                    "raw_guess":  r.get('raw_date') or r['found_date'],
                    "found_date": r['found_date'],
                    "confidence": r['confidence'],
                    "comment":    r.get('comment') or "—",
                    "tags":       r.get('tags'),
                    "raw_date":   r.get('raw_date'),
                    "gps":        list(r['gps']) if r.get('gps') else None,
                    "scene":      r.get('scene'),
                    "setting":    r.get('setting'),
                    "flash":      r.get('flash'),
                })

    if review_queue_accumulator is not None:
        review_queue_accumulator.extend(review_queue)
    else:
        write_review_report(folder, review_queue)

    total          = len(files)
    backs_consumed = len(processed_files)
    reviewed       = len(review_queue)
    date_written   = max(
        0,
        total - backs_consumed - reviewed
        - tagged_no_date_count - nothing_written_count - cutoff_skip_count,
    )

    if batch_totals is not None:
        batch_totals['date_written']    += date_written
        batch_totals['tagged_no_date']  += tagged_no_date_count
        batch_totals['nothing_written'] += nothing_written_count
        batch_totals['reviewed']        += reviewed
        batch_totals['cutoff_skip']     += cutoff_skip_count
        batch_totals['backs']           += backs_consumed
        batch_totals['scanned']         += total

    summary_lines = [f"[bold green]{date_written} dated[/bold green]  of {total} scanned"]
    if tagged_no_date_count:
        summary_lines.append(f"[green]{tagged_no_date_count} tagged (no date)[/green]")
    if backs_consumed:
        summary_lines.append(f"[dim]{backs_consumed} back-of-photo consumed[/dim]")
    if cutoff_skip_count:
        summary_lines.append(f"[dim]{cutoff_skip_count} skipped (after cutoff)[/dim]")
    if reviewed:
        summary_lines.append(f"[yellow]{reviewed} queued for review[/yellow]")
    if nothing_written_count:
        summary_lines.append(f"[red]{nothing_written_count} nothing written[/red]")
    console.print(f"\n  {' · '.join(summary_lines)}")

    return {
        'scanned':         total,
        'date_written':    date_written,
        'tagged_no_date':  tagged_no_date_count,
        'nothing_written': nothing_written_count,
        'reviewed':        reviewed,
        'cutoff_skip':     cutoff_skip_count,
        'backs':           backs_consumed,
    }, _work_pos


def _print_run_complete_panel(bt: dict) -> None:
    """Render the run-complete summary panel from a batch-totals dict."""
    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column(style="dim", width=20)
    tbl.add_column()
    tbl.add_row("Total scanned",  str(bt['scanned']))
    tbl.add_row("[green]Date written[/green]",         f"[green]{bt['date_written']}[/green]")
    if bt.get('tagged_no_date'):
        tbl.add_row("[green]Tagged (no date)[/green]", f"[green]{bt['tagged_no_date']}[/green]")
    if bt.get('backs'):
        tbl.add_row("[dim]Backs consumed[/dim]",       str(bt['backs']))
    if bt.get('cutoff_skip'):
        tbl.add_row("[dim]After cutoff[/dim]",         str(bt['cutoff_skip']))
    if bt.get('reviewed'):
        tbl.add_row("[yellow]Review queue[/yellow]",   f"[yellow]{bt['reviewed']}[/yellow]")
    if bt.get('nothing_written'):
        tbl.add_row("[red]Nothing written[/red]",      f"[red]{bt['nothing_written']}[/red]")
    console.print(Panel(tbl, title="[bold]Run Complete[/bold]", border_style="cyan"))


# ---------------------------------------------------------------------------
# Run summary report
# ---------------------------------------------------------------------------


def _write_run_report(root_folder, run_log):
    """Write a dark-mode HTML summary report for the completed run.

    Parameters
    ----------
    root_folder : str
        Directory where ``metadata-ai-report.html`` is written.
    run_log : dict
        Accumulated run data from ``process_archive``.
    """
    photos  = run_log.get('photos', [])
    folders = run_log.get('folders', [])
    totals  = run_log.get('totals', {})
    settings = run_log.get('settings', {})

    # ── Derived stats ────────────────────────────────────────────────────────
    # Top keywords
    keyword_counts: Counter = Counter()
    for p in photos:
        if p.get('tags'):
            for kw in p['tags'].split(','):
                kw = kw.strip().lower()
                if kw:
                    keyword_counts[kw] += 1
    top_keywords = keyword_counts.most_common(30)

    # Places
    place_counts: Counter = Counter()
    for p in photos:
        if p.get('geo_label'):
            place_counts[p['geo_label'].strip()] += 1
    top_places = place_counts.most_common(20)

    # Decade distribution
    decade_counts: Counter = Counter()
    for p in photos:
        if p.get('decade') is not None:
            decade_counts[p['decade']] += 1
    decades_sorted = sorted(decade_counts.items())

    # Setting breakdown
    indoor  = sum(1 for p in photos if p.get('setting') == 'indoor')
    outdoor = sum(1 for p in photos if p.get('setting') == 'outdoor')

    # Flash breakdown
    flash_yes = sum(1 for p in photos if p.get('flash') == 'yes')
    flash_no  = sum(1 for p in photos if p.get('flash') == 'no')

    # Per-folder table
    folder_stats: dict = {}
    for p in photos:
        fn = p.get('folder', '?')
        if fn not in folder_stats:
            folder_stats[fn] = {'scanned': 0, 'dated': 0, 'tagged': 0}
        folder_stats[fn]['scanned'] += 1
        if p.get('date'):
            folder_stats[fn]['dated'] += 1
        elif p.get('tags') or p.get('scene'):
            folder_stats[fn]['tagged'] += 1

    # Duration
    dur = run_log.get('duration_seconds', 0)
    if dur >= 3600:
        dur_str = f"{dur // 3600}h {(dur % 3600) // 60}m"
    elif dur >= 60:
        dur_str = f"{dur // 60}m {dur % 60}s"
    else:
        dur_str = f"{dur}s"

    # ── HTML helpers ─────────────────────────────────────────────────────────
    def pct_bar(value, total, color='var(--green)'):
        """Inline SVG progress bar."""
        w = round(value / total * 100, 1) if total else 0
        return (
            f'<div class="bar-wrap"><div class="bar-fill" '
            f'style="width:{w}%;background:{color}"></div></div>'
        )

    # ── Sections ─────────────────────────────────────────────────────────────

    # Run summary cards
    total_scanned = totals.get('scanned', len(photos))
    summary_cards = ""
    for label, value, color in [
        ("Scanned",       total_scanned,                    "var(--text)"),
        ("Dated",         totals.get('date_written', 0),    "var(--green)"),
        ("Tagged",        totals.get('tagged_no_date', 0),  "var(--teal)"),
        ("Review queue",  totals.get('reviewed', 0),        "var(--amber)"),
        ("Nothing written", totals.get('nothing_written',0),"var(--red)"),
        ("After cutoff",  totals.get('cutoff_skip', 0),     "var(--dim)"),
    ]:
        summary_cards += (
            f'<div class="stat-card">'
            f'<div class="stat-num" style="color:{color}">{value}</div>'
            f'<div class="stat-lbl">{label}</div>'
            f'</div>'
        )

    # Decade chart
    decade_html = ""
    if decades_sorted:
        max_d = max(v for _, v in decades_sorted) or 1
        for decade, count in decades_sorted:
            w = round(count / max_d * 100, 1)
            decade_html += (
                f'<div class="chart-row">'
                f'<div class="chart-label">{decade}s</div>'
                f'<div class="chart-bar-wrap">'
                f'<div class="chart-bar" style="width:{w}%">{count}</div>'
                f'</div></div>'
            )
    else:
        decade_html = '<p class="empty">No dated photos recorded.</p>'

    # Keyword cloud
    kw_html = ""
    if top_keywords:
        max_kw = top_keywords[0][1] or 1
        for kw, count in top_keywords:
            size = round(0.75 + (count / max_kw) * 1.25, 2)
            opacity = round(0.55 + (count / max_kw) * 0.45, 2)
            kw_html += (
                f'<span class="kw-tag" style="font-size:{size}em;opacity:{opacity}" '
                f'title="{count} photos">{_xml_escape(kw)} '
                f'<span class="kw-count">{count}</span></span>'
            )
    else:
        kw_html = '<p class="empty">No keywords recorded.</p>'

    # Places list
    places_html = ""
    if top_places:
        for place, count in top_places:
            bar = pct_bar(count, top_places[0][1], 'var(--teal)')
            map_url = f"https://www.openstreetmap.org/search?query={place.replace(' ', '+')}"
            places_html += (
                f'<div class="place-row">'
                f'<a class="place-name" href="{map_url}" target="_blank">{_xml_escape(place)}</a>'
                f'{bar}'
                f'<span class="place-count">{count}</span>'
                f'</div>'
            )
    else:
        places_html = '<p class="empty">No locations identified.</p>'

    # Setting / flash mini-stats
    setting_html = ""
    total_setting = indoor + outdoor
    if total_setting:
        setting_html = (
            f'<div class="mini-stat"><span class="ms-label">Indoor</span>'
            f'<span class="ms-val">{indoor}</span>{pct_bar(indoor, total_setting)}</div>'
            f'<div class="mini-stat"><span class="ms-label">Outdoor</span>'
            f'<span class="ms-val">{outdoor}</span>{pct_bar(outdoor, total_setting, "var(--teal)")}</div>'
        )
    total_flash = flash_yes + flash_no
    if total_flash:
        setting_html += (
            f'<div class="mini-stat"><span class="ms-label">Flash used</span>'
            f'<span class="ms-val">{flash_yes}</span>{pct_bar(flash_yes, total_flash)}</div>'
            f'<div class="mini-stat"><span class="ms-label">No flash</span>'
            f'<span class="ms-val">{flash_no}</span>{pct_bar(flash_no, total_flash, "var(--teal)")}</div>'
        )
    if not setting_html:
        setting_html = '<p class="empty">No setting data recorded.</p>'

    # Per-folder table
    folder_rows = ""
    for fname, stats in sorted(folder_stats.items()):
        pct = round(stats['dated'] / stats['scanned'] * 100) if stats['scanned'] else 0
        folder_rows += (
            f'<tr><td class="td-name">{_xml_escape(fname)}</td>'
            f'<td class="td-num">{stats["scanned"]}</td>'
            f'<td class="td-num green">{stats["dated"]}</td>'
            f'<td class="td-num teal">{stats["tagged"]}</td>'
            f'<td class="td-pct">{pct}%</td></tr>'
        )
    folder_table_html = (
        f'<table class="folder-table"><thead>'
        f'<tr><th>Folder</th><th>Scanned</th><th>Dated</th><th>Tagged</th><th>% dated</th></tr>'
        f'</thead><tbody>{folder_rows}</tbody></table>'
        if folder_rows else '<p class="empty">No folder data.</p>'
    )

    # Settings summary
    s = settings
    opt_parts = []
    if s.get('recursive'):        opt_parts.append('recursive')
    if s.get('folder_consensus'): opt_parts.append('consensus')
    if s.get('xmp_only'):         opt_parts.append('xmp-only')
    if s.get('skip_dated'):       opt_parts.append('skip-dated')
    if run_log.get('dry_run'):    opt_parts.append('dry-run')
    opts_html = ' '.join(f'<span class="tag">{o}</span>' for o in opt_parts) or '<span class="dim">none</span>'
    geo_line_html = (
        f'<div class="setting-row"><span>Geo confidence</span>'
        f'<span>≥{s.get("geo_confidence_threshold", 7)}/10</span></div>'
    ) if s.get('enable_geo') else ''

    settings_html = (
        f'<div class="settings-grid">'
        f'<div class="setting-row"><span>Model</span><span>{_xml_escape(run_log.get("model",""))}</span></div>'
        f'<div class="setting-row"><span>Cutoff year</span><span>before {s.get("cutoff_year","")}</span></div>'
        f'<div class="setting-row"><span>Date confidence</span><span>≥{s.get("confidence_threshold","")}/10</span></div>'
        f'<div class="setting-row"><span>Geotagging</span><span>{"yes" if s.get("enable_geo") else "no"}</span></div>'
        f'{geo_line_html}'
        f'<div class="setting-row"><span>Options</span><span>{opts_html}</span></div>'
        f'<div class="setting-row"><span>Started</span><span>{run_log.get("started","")}</span></div>'
        f'<div class="setting-row"><span>Finished</span><span>{run_log.get("finished","")}</span></div>'
        f'<div class="setting-row"><span>Duration</span><span>{dur_str}</span></div>'
        f'</div>'
    )

    # ── Full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Metadata-AI — Run Report</title>
  <style>
    :root {{
      --bg:     #0d0f14;
      --panel:  #161a22;
      --panel2: #1c2030;
      --border: #252b38;
      --text:   #dde1ea;
      --dim:    #7a8494;
      --green:  #4ecb8d;
      --teal:   #3db8c8;
      --amber:  #f0a04a;
      --red:    #e05c5c;
      --accent: #5b9dff;
      --shadow: 0 4px 20px rgba(0,0,0,.45);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; font-size: 14px; line-height: 1.6; padding: 40px; max-width: 1300px; margin: 0 auto; }}
    h1 {{ font-size: 24px; font-weight: 700; letter-spacing: -.02em; }}
    h2 {{ font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: var(--dim); margin-bottom: 14px; }}
    .subtitle {{ color: var(--dim); font-size: 13px; margin-top: 4px; }}
    .header {{ margin-bottom: 36px; padding-bottom: 24px; border-bottom: 1px solid var(--border); }}
    .section {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 24px; margin-bottom: 20px; box-shadow: var(--shadow); }}
    .section-header {{ margin-bottom: 18px; }}
    .stats-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
    .stat-card {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 20px; min-width: 110px; }}
    .stat-num {{ font-size: 26px; font-weight: 700; line-height: 1.1; }}
    .stat-lbl {{ font-size: 11px; color: var(--dim); text-transform: uppercase; letter-spacing: .05em; margin-top: 3px; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
    /* Decade chart */
    .chart-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }}
    .chart-label {{ width: 52px; text-align: right; color: var(--dim); font-size: 12px; flex-shrink: 0; }}
    .chart-bar-wrap {{ flex: 1; background: var(--panel2); border-radius: 4px; height: 22px; overflow: hidden; }}
    .chart-bar {{ height: 100%; background: var(--accent); border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 11px; color: var(--bg); font-weight: 600; min-width: 28px; transition: width .3s; }}
    /* Keyword cloud */
    .kw-cloud {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .kw-tag {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 20px; padding: 3px 12px; cursor: default; white-space: nowrap; }}
    .kw-count {{ font-size: .7em; color: var(--dim); }}
    /* Places */
    .place-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
    .place-name {{ color: var(--teal); text-decoration: none; min-width: 180px; }}
    .place-name:hover {{ text-decoration: underline; }}
    .place-count {{ color: var(--dim); font-size: 12px; min-width: 24px; text-align: right; }}
    /* Progress bar */
    .bar-wrap {{ flex: 1; background: var(--panel2); border-radius: 3px; height: 8px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 3px; }}
    /* Mini stats */
    .mini-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .mini-stat {{ display: flex; align-items: center; gap: 8px; }}
    .ms-label {{ color: var(--dim); font-size: 12px; min-width: 80px; }}
    .ms-val {{ font-weight: 600; min-width: 28px; text-align: right; }}
    /* Folder table */
    .folder-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .folder-table th {{ text-align: left; color: var(--dim); font-weight: 500; padding: 6px 10px; border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }}
    .folder-table td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); }}
    .folder-table tr:last-child td {{ border-bottom: none; }}
    .td-name {{ color: var(--text); }}
    .td-num {{ text-align: center; }}
    .td-pct {{ text-align: center; color: var(--dim); }}
    .green {{ color: var(--green); }}
    .teal  {{ color: var(--teal);  }}
    /* Settings */
    .settings-grid {{ display: grid; gap: 6px; }}
    .setting-row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 13px; }}
    .setting-row:last-child {{ border-bottom: none; }}
    .setting-row span:first-child {{ color: var(--dim); }}
    .tag {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 4px; padding: 1px 7px; font-size: 11px; }}
    .dim {{ color: var(--dim); }}
    .empty {{ color: var(--dim); font-style: italic; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Metadata-AI — Run Report</h1>
    <div class="subtitle">{_xml_escape(root_folder)} &nbsp;·&nbsp; {_xml_escape(run_log.get('started',''))}</div>
  </div>

  <div class="stats-row">{summary_cards}</div>

  <div class="two-col">

    <div class="section">
      <div class="section-header"><h2>Photos by Decade</h2></div>
      {decade_html}
    </div>

    <div class="section">
      <div class="section-header"><h2>Setting &amp; Flash</h2></div>
      <div class="mini-stats">{setting_html}</div>
    </div>

  </div>

  <div class="section">
    <div class="section-header"><h2>Top Keywords</h2></div>
    <div class="kw-cloud">{kw_html}</div>
  </div>

  <div class="section">
    <div class="section-header"><h2>Places Identified</h2></div>
    {places_html}
  </div>

  <div class="section">
    <div class="section-header"><h2>Folders Processed</h2></div>
    {folder_table_html}
  </div>

  <div class="section">
    <div class="section-header"><h2>Run Settings</h2></div>
    {settings_html}
  </div>

</body>
</html>"""

    report_path = os.path.join(root_folder, "metadata-ai-report.html")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        console.print(f"\n[bold green]✓ Run report:[/bold green] {report_path}")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not write run report: {e}[/yellow]")


# ---------------------------------------------------------------------------
# Archive orchestrator
# ---------------------------------------------------------------------------


def process_archive(
    folder,
    cutoff_year=2010,
    confidence_threshold=7,
    xmp_only=False,
    enable_geo=False,
    recursive=False,
    folder_consensus=False,
    dry_run=False,
    skip_dated=False,
    geo_confidence_threshold=7,
):
    """Process a directory of photos, optionally walking all subfolders.

    Parameters
    ----------
    folder : str
        Root directory to process.
    cutoff_year : int, optional
        Photos dated from this year or later are skipped.  Default 2010.
    confidence_threshold : int, optional
        Minimum VLM date confidence to write without review.  Default 7.
    xmp_only : bool, optional
        Write XMP sidecars only.  Default ``False``.
    enable_geo : bool, optional
        Enable Nominatim geotagging.  Default ``False``.
    recursive : bool, optional
        Walk all subfolders when ``True``.  Default ``False``.
    folder_consensus : bool, optional
        Apply consensus-year correction after each folder.  Default ``False``.
    dry_run : bool, optional
        Log actions without modifying files.  Default ``False``.
    skip_dated : bool, optional
        Skip photos with an existing ``DateTimeOriginal`` tag.  Default ``False``.
    geo_confidence_threshold : int, optional
        Minimum VLM location confidence for geocoding.  Default 7.

    Returns
    -------
    review_folder : str or None
        Directory containing ``review.json`` when items need review;
        ``None`` otherwise.
    counters : dict
        Batch-totals dict (same keys as ``_process_folder`` return value).

    Notes
    -----
    In recursive mode, each subfolder is re-scanned from disk immediately
    before processing so files added while the run is in progress are
    automatically included.  New subfolders appearing mid-run are also
    picked up on each discovery loop iteration.
    """
    if not os.path.exists(folder):
        console.print(f"[red]Directory not found: {folder}[/red]")
        return None, {}

    start_pause_listener()

    run_log = {
        'root':       folder,
        'model':      MODEL_ID,
        'started':    datetime.now().isoformat(timespec='seconds'),
        'finished':   None,
        'dry_run':    dry_run,
        'folders':    [],
        'photos':     [],
        'settings': {
            'cutoff_year':            cutoff_year,
            'confidence_threshold':   confidence_threshold,
            'geo_confidence_threshold': geo_confidence_threshold,
            'xmp_only':               xmp_only,
            'enable_geo':             enable_geo,
            'recursive':              recursive,
            'folder_consensus':       folder_consensus,
            'skip_dated':             skip_dated,
        },
    }
    _run_start = time.monotonic()

    flags = []
    if recursive:         flags.append("recursive")
    if enable_geo:        flags.append("geotag")
    if folder_consensus:  flags.append("consensus")
    if xmp_only:          flags.append("xmp-only")
    if dry_run:           flags.append("DRY RUN")
    if skip_dated:        flags.append("skip-dated")
    flags_str = "  ".join(f"[cyan]{f}[/cyan]" for f in flags) if flags else "[dim]none[/dim]"
    geo_line  = f"\n[dim]Geo conf:[/dim]   ≥{geo_confidence_threshold}/10" if enable_geo else ""
    console.print(Panel.fit(
        f"[bold]Metadata-AI[/bold]\n\n"
        f"[dim]Folder:[/dim]     {folder}\n"
        f"[dim]Model:[/dim]      {MODEL_ID}\n"
        f"[dim]Cutoff:[/dim]     before {cutoff_year}\n"
        f"[dim]Date conf:[/dim]   ≥{confidence_threshold}/10"
        f"{geo_line}\n"
        f"[dim]Options:[/dim]    {flags_str}\n"
        f"[dim]Tip:[/dim]        type [bold]p[/bold] + Enter to pause between photos",
        border_style="cyan",
        title="[bold]Metadata-AI[/bold]",
    ))

    if recursive:
        seen_subfolders: list[str] = []
        seen_set: set[str]         = set()
        initial_count              = 0
        for root, dirs, filenames in os.walk(folder):
            dirs.sort()
            if root not in seen_set:
                seen_subfolders.append(root)
                seen_set.add(root)
            initial_count += sum(1 for f in filenames if f.lower().endswith(EXTENSIONS))

        console.print(
            f"\n[bold]📂 {initial_count} files across all subfolders[/bold] "
            f"[dim](rescanning each folder before processing)[/dim]"
        )

        accumulated_review_queue = []
        shared_durations         = deque(maxlen=5)
        batch_totals             = {
            'scanned': 0, 'date_written': 0, 'tagged_no_date': 0,
            'nothing_written': 0, 'reviewed': 0, 'cutoff_skip': 0, 'backs': 0,
        }
        global_offset        = 0
        work_offset          = 0   # real (non-skipped) photos processed so far
        processed_subfolders: set[str] = set()

        # Compute work_total once: files not yet in the checkpoint.
        # Used by _process_folder to show [X/work_total] across the whole run.
        _checkpoint = _load_checkpoint(folder) if not dry_run else set()
        _run_work_total = max(initial_count - len(_checkpoint), 1)

        while True:
            for root, dirs, _ in os.walk(folder):
                dirs.sort()
                if root not in seen_set:
                    seen_subfolders.append(root)
                    seen_set.add(root)

            next_subfolder = next(
                (sf for sf in seen_subfolders if sf not in processed_subfolders),
                None,
            )
            if next_subfolder is None:
                break

            subfolder = next_subfolder
            processed_subfolders.add(subfolder)

            try:
                disk_files = natsorted([
                    f for f in os.listdir(subfolder)
                    if f.lower().endswith(EXTENSIONS)
                    and os.path.isfile(os.path.join(subfolder, f))
                ])
            except (PermissionError, FileNotFoundError) as e:
                console.print(f"  [yellow]⚠ Cannot read {subfolder} ({e}) — skipping.[/yellow]")
                continue

            if not disk_files:
                continue

            remaining_known = 0
            for sf in seen_subfolders:
                if sf not in processed_subfolders:
                    try:
                        remaining_known += sum(
                            1 for f in os.listdir(sf)
                            if f.lower().endswith(EXTENSIONS)
                            and os.path.isfile(os.path.join(sf, f))
                        )
                    except (PermissionError, FileNotFoundError):
                        pass
            live_global_total = global_offset + len(disk_files) + remaining_known

            console.print(
                f"\n[bold cyan]📁 {subfolder}[/bold cyan] "
                f"[dim]({len(disk_files)} file(s) on disk)[/dim]"
            )
            _, new_work_pos = _process_folder(
                subfolder, disk_files, cutoff_year, confidence_threshold, xmp_only, enable_geo,
                global_offset=global_offset, global_total=live_global_total,
                folder_consensus=folder_consensus,
                root_folder=folder, dry_run=dry_run, skip_dated=skip_dated,
                review_queue_accumulator=accumulated_review_queue,
                shared_durations=shared_durations, batch_totals=batch_totals,
                geo_confidence_threshold=geo_confidence_threshold,
                run_log=run_log,
                work_offset=work_offset,
                work_total=_run_work_total,
            )
            work_offset    = new_work_pos
            global_offset += len(disk_files)

        write_review_report(folder, accumulated_review_queue)
        _print_run_complete_panel(batch_totals)
        run_log['finished'] = datetime.now().isoformat(timespec='seconds')
        run_log['duration_seconds'] = int(time.monotonic() - _run_start)
        run_log['totals'] = batch_totals
        _write_run_report(folder, run_log)

        if not dry_run:
            _clear_checkpoint(folder)
        review_folder = folder if accumulated_review_queue else None
        return review_folder, batch_totals

    files    = natsorted([f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)])
    _nr_checkpoint  = _load_checkpoint(folder) if not dry_run else set()
    _nr_work_total  = max(len(files) - len(_nr_checkpoint), 1)
    counters, _ = _process_folder(
        folder, files, cutoff_year, confidence_threshold, xmp_only, enable_geo,
        folder_consensus=folder_consensus, dry_run=dry_run, skip_dated=skip_dated,
        geo_confidence_threshold=geo_confidence_threshold,
        run_log=run_log,
        work_offset=0,
        work_total=_nr_work_total,
    )
    _print_run_complete_panel(counters)
    run_log['finished'] = datetime.now().isoformat(timespec='seconds')
    run_log['duration_seconds'] = int(time.monotonic() - _run_start)
    run_log['totals'] = counters
    _write_run_report(folder, run_log)
    if not dry_run:
        _clear_checkpoint(folder)
    review_folder = folder if os.path.exists(_review_json_path(folder)) else None
    return review_folder, counters


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def _video_get_duration(video_path):
    """Return the duration of a video file in seconds."""
    cmd = [
        _resolve_binary("ffprobe"), "-v", "quiet",
        "-print_format", "json", "-show_format", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(result.stdout)["format"]["duration"])


def _video_extract_frames(video_path, interval, out_dir):
    """Extract one JPEG frame every *interval* seconds using ffmpeg."""
    duration = _video_get_duration(video_path)

    effective_interval = interval
    if effective_interval >= duration:
        effective_interval = max(1, int(duration / 2))
        console.print(
            f"  [yellow]⚠ Interval ({interval}s) ≥ video duration ({duration:.1f}s) "
            f"— reduced to {effective_interval}s.[/yellow]"
        )

    pattern = os.path.join(out_dir, "frame_%08d.jpg")
    fps_str = f"1/{int(effective_interval)}"
    cmd = [
        _resolve_binary("ffmpeg"), "-y", "-i", video_path,
        "-vf", f"fps={fps_str}",
        "-q:v", "3",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-500:]}")

    frames      = []
    frame_files = sorted(
        f for f in os.listdir(out_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )
    for idx, fname in enumerate(frame_files):
        ts = idx * effective_interval
        if ts < duration:
            frames.append((ts, os.path.join(out_dir, fname)))
    return frames


def _video_format_ts(seconds):
    """Format *seconds* as an ``HH:MM:SS`` timestamp string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _video_parse_field(text, field):
    """Extract a named field value from a structured VLM response."""
    m   = re.search(rf'^{field}:\s*(.+)', text, re.MULTILINE | re.IGNORECASE)
    val = m.group(1).strip() if m else ""
    val = re.sub(r'[*_`]', '', val).strip()
    return val if val.lower() not in ("unknown", "none", "") else ""


def _video_parse_year(text):
    """Extract the most plausible single year from a date string."""
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
            yr_str = formatter(m)[:4]
            try:
                yr = int(yr_str)
                if MIN_VIDEO_YEAR <= yr <= MAX_YEAR:
                    return yr
            except ValueError:
                pass
    return None


def _video_consensus_year(year_conf_pairs, threshold):
    """Compute the plurality consensus year from frame-level estimates.

    A strict majority (``best_votes * 2 > total``) is required — a 50/50 tie
    is intentionally **not** a majority.  This mirrors the integer-arithmetic
    test used in :func:`_apply_consensus_year` and avoids floating-point
    comparison inconsistencies.
    """
    eligible = [(yr, c) for yr, c in year_conf_pairs if yr and c >= threshold]
    if not eligible:
        return None, 0, 0, False
    counts           = Counter(yr for yr, _ in eligible)
    best, best_votes = counts.most_common(1)[0]
    total            = len(eligible)
    majority         = best_votes * 2 > total   # strict majority; mirrors _apply_consensus_year
    return best, best_votes, total, majority


def _video_analyze_frame(ts, image_path, index, total, video_name=""):
    """Analyse a single video frame and return a description dict."""
    _looks_technical = bool(re.search(
        r'(fps|mbps|kbps|\d+x\d+|bitrate|codec|h264|h265|hevc|avc)',
        video_name, re.IGNORECASE,
    ))
    context_hint = (
        f"\nContext: this frame is from a video file named '{video_name}'. "
        "Treat this as high-confidence date and location information."
    ) if video_name and not _looks_technical else ""

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
    """Synthesise all frame descriptions into a cohesive video summary."""
    all_descriptions = [
        f"[{_video_format_ts(ts)}]\n{d['description']}"
        for ts, d in frame_analyses if d.get("description")
    ]
    MAX_CHARS = 24000
    selected  = all_descriptions
    while len(selected) > 1 and sum(len(s) for s in selected) > MAX_CHARS:
        selected = selected[::2]
    if len(selected) < len(all_descriptions):
        console.print(
            f"      [dim](Summarizing {len(selected)} of {len(all_descriptions)} "
            f"frames to fit context window)[/dim]"
        )

    frame_descriptions = "\n\n".join(selected)
    prompt             = VIDEO_SUMMARY_PROMPT.format(
        interval=interval, frame_descriptions=frame_descriptions,
    )
    with console.status("[dim]Synthesizing summary…[/dim]", spinner="dots"):
        try:
            resp   = CLIENT.chat.completions.create(
                model=MODEL_ID, max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            result        = resp.choices[0].message.content.strip()
            finish_reason = getattr(resp.choices[0], 'finish_reason', None)
            ends_mid      = result and not result.rstrip().endswith(('.', '!', '?', '"', ')', ']'))

            if (finish_reason == "length" or ends_mid) and len(selected) > 1:
                selected           = selected[::2]
                frame_descriptions = "\n\n".join(selected)
                prompt2            = VIDEO_SUMMARY_PROMPT.format(
                    interval=interval, frame_descriptions=frame_descriptions,
                )
                resp2         = CLIENT.chat.completions.create(
                    model=MODEL_ID, max_tokens=4096,
                    messages=[{"role": "user", "content": prompt2}],
                )
                result        = resp2.choices[0].message.content.strip()
                retry_finish  = getattr(resp2.choices[0], 'finish_reason', None)
                if retry_finish == "length":
                    console.print("[yellow]⚠ Summary may be incomplete (truncated after retry).[/yellow]")

            return result
        except Exception as e:
            console.print(f"[red]✗ Summary generation failed: {e}[/red]")
            return "(Summary generation failed.)"


def _video_extract_metadata(summary, consensus_yr):
    """Ask the VLM to extract structured metadata from a video summary."""
    with console.status("[dim]Extracting metadata fields…[/dim]", spinner="dots"):
        prompt = VIDEO_METADATA_PROMPT.format(summary=summary)
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
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
    """Write metadata into the video container using ffmpeg stream copy."""
    tmp_path  = video_path + ".tmp" + Path(video_path).suffix
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

    cmd = [_resolve_binary("ffmpeg"), "-y", "-i", video_path, "-c", "copy", *meta_args, tmp_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"   [yellow]⚠ ffmpeg error: {result.stderr[-300:]}[/yellow]")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
        os.replace(tmp_path, video_path)
        return True
    except Exception as e:
        console.print(f"   [yellow]⚠ ffmpeg exception: {e}[/yellow]")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def _video_build_report(video_path, interval, frame_analyses, summary, consensus_yr):
    """Build a plain-text analysis report for a video."""
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
        note = (
            f"  [date: {raw} → {yr}, confidence: {conf}/10]" if yr
            else f"  [date: {raw}, confidence: {conf}/10]"
        )
        lines.append(f"\n[{_video_format_ts(ts)}]{note}")
        lines.append(desc)
    lines += ["", "=" * 72]
    return "\n".join(lines)


def process_video(video_path, interval=30, output_path=None):
    """Run the full video analysis pipeline for a single file.

    Parameters
    ----------
    video_path : str
        Absolute path to the video file.
    interval : int, optional
        Target seconds between sampled frames.  Default 30.
    output_path : str or None, optional
        Destination for the plain-text report.  Defaults to
        ``<video_stem>_summary.txt`` in the same directory.

    Notes
    -----
    The pipeline: extract frames → analyse each frame → compute consensus
    year → synthesise summary → extract structured metadata → offer the
    user a preview and optional edit → write metadata into the container →
    save the text report.
    """
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
        video_stem     = Path(video_path).stem
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
                yr_str     = str(data["year"]) if data["year"] else "?"
                flag       = "✓" if data["confidence"] >= 6 and data["year"] else "~"
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
            "  python metadata-ai.py /path/to/directory --cutoff 1995 --date-confidence 6 --xmp-only"
        ),
    )
    parser.add_argument("directory", nargs="?", default=None,
                        help="Path to directory containing photos or videos (prompted if omitted)")
    parser.add_argument("--cutoff", type=int, default=None, metavar="YEAR",
                        help="Skip photos dated from this year or later (default: 2010)")
    parser.add_argument("--date-confidence", type=int, default=None, metavar="1-10",
                        help="Minimum date confidence required to write without review (default: 7)")
    parser.add_argument("--xmp-only", action="store_true", default=None,
                        help="Write metadata to XMP sidecar files only")
    parser.add_argument("--geotag", action="store_true", default=None,
                        help="Enable geotagging via Nominatim")
    parser.add_argument("--geo-confidence", type=int, default=None, metavar="1-10",
                        help="Minimum VLM location confidence required before geocoding (default: 7). Higher = stricter.")
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

    if args.model:
        MODEL_ID = args.model

    # ── Welcome header ───────────────────────────────────────────────────────
    # Shown once at startup so the user knows what they're configuring.
    # Only displayed in interactive mode — CLI-only runs (all args supplied
    # via flags) skip straight to process_archive's own run panel.
    if sys.stdin.isatty():
        console.print()
        console.print(Panel(
            f"\n"
            f"  [bold white]Metadata-AI[/bold white]\n"
            f"  [dim]Automatic photo tagging and dating via local AI[/dim]\n\n"
            f"  [dim]Model:[/dim]  [cyan]{MODEL_ID}[/cyan]\n"
            f"  [dim]Server:[/dim] [cyan]LM Studio  ·  localhost:1234[/cyan]\n",
            border_style="cyan",
            padding=(0, 2),
        ))
        console.print()

    # _strip_shell_escapes is defined at module level (Platform helpers section).
    # The _is_windows local is no longer needed here.

    if args.directory:
        input_path = _strip_shell_escapes(args.directory.strip())
    else:
        input_path = _strip_shell_escapes(_ask("Directory or file path", default=DIRECTORY)) or DIRECTORY

    if args.review:
        if not os.path.isdir(input_path):
            console.print(
                f"[red]--review expects a directory containing review.json. Got: {input_path}[/red]"
            )
            sys.exit(1)
        xmp_only = args.xmp_only or False
        run_review_pass(input_path, xmp_only=xmp_only)
        sys.exit(0)

    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            if args.video_interval is not None:
                interval = args.video_interval
            else:
                raw = _ask("Seconds between frames", "30")
                try:
                    interval = int(raw)
                except ValueError:
                    interval = 30
            process_video(input_path, interval, output_path=getattr(args, 'output', None))
        elif ext in EXTENSIONS:
            folder   = os.path.dirname(input_path) or "."
            filename = os.path.basename(input_path)
            _process_folder(
                folder, [filename],
                cutoff_year          = args.cutoff if args.cutoff is not None else 2010,
                confidence_threshold = args.date_confidence if args.date_confidence is not None else 7,
                xmp_only             = args.xmp_only or False,
                enable_geo           = args.geotag or False,
                folder_consensus     = args.consensus or False,
                dry_run              = args.dry_run,
                skip_dated           = args.skip_dated,
                work_offset          = 0,
                work_total           = 1,
            )
        else:
            console.print(f"[red]Unsupported file type: {ext}[/red]")
        sys.exit(0)

    directory = input_path

    if _yn_select("Analyze video files in this directory?"):
        video_files = natsorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ])
        if not video_files:
            console.print("[dim]No video files found — switching to photo mode.[/dim]")
        else:
            raw = _ask("Seconds between frames", "30")
            try:
                video_interval = int(raw)
            except ValueError:
                video_interval = 30
            if not args.model:
                model_input = _ask("LM Studio model", MODEL_ID)
                if model_input and model_input != MODEL_ID:
                    MODEL_ID = model_input
                    console.print(f"[dim]Using model: {MODEL_ID}[/dim]")
            console.print(f"\n[bold]{len(video_files)} video file(s) found.[/bold]")
            for vf in video_files:
                console.print(f"\n[bold cyan]📹 {vf}[/bold cyan]")
                process_video(os.path.join(directory, vf), video_interval)
            sys.exit(0)

    directories = [directory]
    if sys.stdin.isatty():
        while _yn_select("Add another folder to this batch?", default_yes=False):
            extra = _ask("Path to additional folder", "")
            extra = _strip_shell_escapes(extra.strip()) if extra else ""
            if not extra:
                continue
            if not os.path.isdir(extra):
                console.print(f"  [yellow]⚠ Not a directory — ignored: {extra}[/yellow]")
                continue
            if extra in directories:
                console.print(f"  [dim]Already in batch — ignored: {extra}[/dim]")
                continue
            directories.append(extra)
            console.print(
                f"  [green]✓[/green] Added: {extra}  "
                f"[dim]({len(directories)} folder(s) in batch)[/dim]"
            )
        if len(directories) > 1:
            console.print(
                f"\n[bold]Batch:[/bold] [cyan]{len(directories)}[/cyan] folder(s) "
                f"— settings below will apply to all of them.\n"
            )

    if args.cutoff is not None:
        cutoff_year = args.cutoff
    else:
        raw = _ask("Skip photos dated from which year or later?", str(_CONFIG["defaults"]["cutoff_year"]))
        try:
            cutoff_year = int(raw)
        except ValueError:
            console.print("[dim]Invalid year — defaulting to 2010.[/dim]")
            cutoff_year = 2010

    if args.date_confidence is not None:
        confidence_threshold = args.date_confidence
    else:
        raw = _ask("Date confidence threshold (1-10)", str(_CONFIG["defaults"]["date_confidence"]))
        try:
            confidence_threshold = int(raw)
        except ValueError:
            confidence_threshold = 7

    if args.xmp_only is not None:
        xmp_only = args.xmp_only
    else:
        xmp_only = _yn_select("Write metadata to XMP sidecar files only?")

    if args.geotag is not None:
        enable_geo = args.geotag
    else:
        enable_geo = _yn_select("Enable geotagging?")

    if enable_geo:
        if args.geo_confidence is not None:
            geo_confidence_threshold = args.geo_confidence
        else:
            raw = _ask("Location confidence threshold (1-10, higher = stricter)", str(_CONFIG["defaults"]["geo_confidence"]))
            try:
                geo_confidence_threshold = max(1, min(10, int(raw)))
            except ValueError:
                geo_confidence_threshold = 7
    else:
        geo_confidence_threshold = 7

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

    # ── Configuration summary panel ─────────────────────────────────────────
    console.rule("[dim]Configuration[/dim]")
    cfg_tbl = Table(show_header=False, box=None, padding=(0, 2))
    cfg_tbl.add_column(style="dim", width=18)
    cfg_tbl.add_column()

    folder_display = "\n".join(f"[cyan]{d}[/cyan]" for d in directories)
    cfg_tbl.add_row("Folder(s)", folder_display)
    cfg_tbl.add_row("Model", MODEL_ID)
    cfg_tbl.add_row("Cutoff year", f"before {cutoff_year}")
    cfg_tbl.add_row("Date confidence", f"≥{confidence_threshold}/10")

    if enable_geo:
        cfg_tbl.add_row("Geotagging", f"[green]yes[/green]  [dim](location conf ≥{geo_confidence_threshold}/10)[/dim]")
    else:
        cfg_tbl.add_row("Geotagging", "[dim]no[/dim]")

    flags = []
    if recursive:        flags.append("recursive")
    if folder_consensus: flags.append("consensus")
    if xmp_only:         flags.append("xmp-only")
    if args.skip_dated:  flags.append("skip-dated")
    if args.dry_run:     flags.append("[bold yellow]DRY RUN[/bold yellow]")
    cfg_tbl.add_row("Options", "  ".join(f"[cyan]{f}[/cyan]" for f in flags) if flags else "[dim]none[/dim]")
    cfg_tbl.add_row("Tip", "type [bold]p[/bold] + Enter to pause between photos")

    console.print(Panel(
        cfg_tbl,
        title="[bold]Configuration[/bold]",
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print()

    if not args.dry_run:
        for d in directories:
            progress_file = _checkpoint_path(d)
            if os.path.exists(progress_file):
                with open(progress_file, encoding='utf-8') as pf:
                    completed_count = sum(1 for line in pf if line.strip())
                hint = f" ({d})" if len(directories) > 1 else ""
                if not _yn(
                    f"Found a previous session with {completed_count} completed file(s){hint}. Resume?",
                    default_yes=True,
                ):
                    _clear_checkpoint(d)
                    console.print(f"[dim]Starting fresh for {d}.[/dim]\n")
                else:
                    console.print(f"[dim]Resuming previous session{hint}.[/dim]\n")

    review_folders = []
    all_counters   = []
    for d in directories:
        review_folder, counters = process_archive(
            d, cutoff_year, confidence_threshold, xmp_only, enable_geo, recursive,
            folder_consensus, dry_run=args.dry_run, skip_dated=args.skip_dated,
            geo_confidence_threshold=geo_confidence_threshold,
        )
        if review_folder:
            review_folders.append(review_folder)
        all_counters.append(counters)

    if len(directories) > 1:
        combined = {
            k: sum(c.get(k, 0) for c in all_counters)
            for k in (
                'scanned', 'date_written', 'tagged_no_date',
                'nothing_written', 'reviewed', 'cutoff_skip', 'backs',
            )
        }
        console.print()
        _print_run_complete_panel(combined)

    if review_folders and not args.dry_run:
        total_pending = 0
        for rf in review_folders:
            data = _load_review_json(rf)
            if data:
                total_pending += sum(
                    1 for i in data.get("items", []) if i.get("status") == "pending"
                )
        if total_pending:
            label = (
                f"[cyan]{total_pending}[/cyan] photo(s) need review across "
                f"[cyan]{len(review_folders)}[/cyan] folder(s)"
                if len(review_folders) > 1
                else f"[cyan]{total_pending}[/cyan] photo(s) need review"
            )
            if _yn(f"{label}. Run interactive review now?"):
                for rf in review_folders:
                    if len(review_folders) > 1:
                        console.print(f"\n[bold cyan]📁 {rf}[/bold cyan]")
                    run_review_pass(rf, xmp_only=xmp_only)
            else:
                if len(review_folders) == 1:
                    console.print(
                        f"[dim]Run later with:[/dim] [cyan]python metadata-ai.py "
                        f"{_quote_path(review_folders[0])} --review[/cyan]"
                    )
                else:
                    console.print(f"[dim]Run later with --review on each folder:[/dim]")
                    for rf in review_folders:
                        console.print(
                            f"  [cyan]python metadata-ai.py {_quote_path(rf)} --review[/cyan]"
                        )
