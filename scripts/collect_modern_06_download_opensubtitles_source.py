#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collect_16_download_opensubtitles_source.py

Purpose:
- Download OpenSubtitles English raw text from OPUS
- Lightly normalize and clean the raw subtitle lines
- Save a reusable local text file for later candidate extraction

Outputs:
- data/raw/opensubtitles/opensubtitles_en_raw.txt
- data/raw/opensubtitles/opensubtitles_download_stats.json

Recommended usage:
    python collect_16_download_opensubtitles_source.py

Optional:
    python collect_16_download_opensubtitles_source.py --max-lines 500000
    python collect_16_download_opensubtitles_source.py --force-download
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path
import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
OPEN_SUB_DIR = DATA_RAW_DIR / "opensubtitles"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
OPEN_SUB_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OUTPUT_FILE = OPEN_SUB_DIR / "opensubtitles_en_raw.txt"
DEFAULT_STATS_FILE = OPEN_SUB_DIR / "opensubtitles_download_stats.json"
DEFAULT_ARCHIVE_FILE = OPEN_SUB_DIR / "OpenSubtitles_latest_en.txt.gz"


# ============================================================
# Source config
# ============================================================

# OPUS hosts OpenSubtitles as a downloadable corpus collection.
# This direct English text file path is kept explicit here so the
# download stage stays reproducible and separate from later extraction.
DEFAULT_DOWNLOAD_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/en.txt.gz"


# ============================================================
# Cleaning config
# ============================================================

MULTISPACE_RE = re.compile(r"\s+")
LETTER_RE = re.compile(r"[A-Za-z]")
TIMESTAMP_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:,\d{1,3})?\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?(?:,\d{1,3})?\b"
)
TIME_ONLY_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?(?:,\d{1,3})?\s*$")
BRACKET_NOISE_RE = re.compile(r"\[(.*?)\]|\((.*?)\)|\{(.*?)\}")
SPEAKER_PREFIX_RE = re.compile(r"^\s*[A-Z][A-Z0-9 _-]{1,25}:\s+")
LEADING_DASH_RE = re.compile(r"^\s*[-–—]+\s*")
NON_TEXT_LINE_RE = re.compile(r"^\s*\d+\s*$")


# ============================================================
# Helpers
# ============================================================

def norm(text: str) -> str:
    if text is None:
        return ""
    return MULTISPACE_RE.sub(" ", str(text).strip())


def normalize_line(text: str) -> str:
    """
    Light subtitle-line normalization for raw storage.
    """
    text = norm(text)

    if not text:
        return ""

    text = TIMESTAMP_RE.sub(" ", text)
    text = BRACKET_NOISE_RE.sub(" ", text)
    text = SPEAKER_PREFIX_RE.sub("", text)
    text = LEADING_DASH_RE.sub("", text)

    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
    )

    text = norm(text)

    if not text:
        return ""

    if NON_TEXT_LINE_RE.match(text):
        return ""

    if TIME_ONLY_RE.match(text):
        return ""

    return text


def looks_like_good_raw_line(text: str, min_line_length: int) -> bool:
    """
    Keep only useful English subtitle/dialogue lines.
    """
    text = norm(text)

    if not text:
        return False

    if len(text) < min_line_length:
        return False

    if len(text) > 250:
        return False

    if not LETTER_RE.search(text):
        return False

    token_count = len(text.split())
    if token_count < 3 or token_count > 40:
        return False

    return True


def download_file(url: str, out_path: Path, force_download: bool = False) -> Path:
    """
    Download a remote file unless it already exists.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force_download:
        print(f"[INFO] Using existing archive: {out_path}")
        return out_path

    print(f"[INFO] Downloading: {url}")
    print(f"[INFO] Saving archive to: {out_path}")

    with urllib.request.urlopen(url, context=ssl_context) as response, open(out_path, "wb") as f:
        shutil.copyfileobj(response, f)

    return out_path


def iter_lines_from_archive(archive_path: Path):
    """
    Stream text lines from a .gz or .txt file.
    """
    archive_path = Path(archive_path)
    suffix = archive_path.suffix.lower()

    if suffix == ".gz":
        with gzip.open(archive_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
    elif suffix == ".txt":
        with open(archive_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
    else:
        raise ValueError(f"Unsupported archive type: {archive_path.suffix}")


# ============================================================
# Main
# ============================================================

def download_opensubtitles_source(
    download_url: str = DEFAULT_DOWNLOAD_URL,
    archive_file: Path = DEFAULT_ARCHIVE_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
    stats_file: Path = DEFAULT_STATS_FILE,
    max_lines: int = 500_000,
    min_line_length: int = 8,
    force_download: bool = False,
):
    """
    Download and prepare OpenSubtitles English raw dialogue lines from OPUS.
    """
    archive_file = Path(archive_file)
    output_file = Path(output_file)
    stats_file = Path(stats_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    archive_path = download_file(
        url=download_url,
        out_path=archive_file,
        force_download=force_download,
    )

    total_lines_read = 0
    total_lines_kept = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for raw_line in iter_lines_from_archive(archive_path):
            total_lines_read += 1

            line = normalize_line(raw_line)
            if not looks_like_good_raw_line(line, min_line_length=min_line_length):
                continue

            out_f.write(line + "\n")
            total_lines_kept += 1

            if max_lines > 0 and total_lines_kept >= max_lines:
                break

    stats = {
        "download_url": download_url,
        "archive_file": str(archive_path),
        "output_file": str(output_file),
        "max_lines_requested": int(max_lines),
        "min_line_length": int(min_line_length),
        "total_lines_read": int(total_lines_read),
        "total_lines_kept": int(total_lines_kept),
    }

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Saved raw text file:", output_file)
    print("Saved stats file:", stats_file)
    print("Total lines read:", total_lines_read)
    print("Total lines kept:", total_lines_kept)

    return output_file, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare OpenSubtitles English raw text from OPUS."
    )
    parser.add_argument("--download-url", type=str, default=DEFAULT_DOWNLOAD_URL)
    parser.add_argument("--archive-file", type=str, default=str(DEFAULT_ARCHIVE_FILE))
    parser.add_argument("--output-file", type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--stats-file", type=str, default=str(DEFAULT_STATS_FILE))
    parser.add_argument("--max-lines", type=int, default=500_000)
    parser.add_argument("--min-line-length", type=int, default=8)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    download_opensubtitles_source(
        download_url=args.download_url,
        archive_file=Path(args.archive_file),
        output_file=Path(args.output_file),
        stats_file=Path(args.stats_file),
        max_lines=args.max_lines,
        min_line_length=args.min_line_length,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()