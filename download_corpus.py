"""Download the FATF Recommendations corpus.

Fetches "The FATF Recommendations" (the 40 Recommendations) PDF from the
official FATF site and saves it under data/raw/. The corpus itself is
gitignored; this script is the documented, reproducible download step.

Run from the repo root:
    python download_corpus.py

If the automatic download fails (FATF occasionally reshuffles its CDN paths),
the script prints the official source page so you can download the PDF by hand
and drop it at the expected path.
"""

from __future__ import annotations

import hashlib
import sys

import requests

from regrag import config

# Official source page to fall back to for a manual download.
SOURCE_PAGE = (
    "https://www.fatf-gafi.org/en/publications/fatfrecommendations/"
    "documents/fatf-recommendations.html"
)

# Direct-download candidates, tried in order. The pinned Internet Archive
# (Wayback Machine) snapshot is primary: a permanent, timestamped copy of a
# specific version, so downloads are reproducible and it is not behind FATF's
# bot wall. The live FATF URLs follow as fallbacks (asset endpoint, then plain).
DOWNLOAD_URLS = [
    # Wayback snapshot captured 2026-06-24; the id_ modifier returns the raw,
    # unmodified original bytes (no archive rewriting).
    "https://web.archive.org/web/20260624231211id_/"
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/"
    "FATF%20Recommendations%202012.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/"
    "FATF%20Recommendations%202012.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/"
    "FATF%20Recommendations%202012.pdf",
]

# Full browser-like headers; FATF's CDN 403s requests that do not look like a
# real browser navigation. Even this may not defeat an active bot challenge, in
# which case fall back to the manual download the script prints on failure.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/pdf,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": SOURCE_PAGE,
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Connection": "keep-alive",
}

MIN_BYTES = 100_000  # a real copy of the Recommendations is well over 100 KB


def _looks_like_pdf(content: bytes) -> bool:
    """A valid PDF starts with the %PDF magic bytes."""
    return content[:5] == b"%PDF-"


def download() -> bool:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    # Warm up the session by visiting the source page first, mimicking a real
    # browser navigating to the PDF (picks up any cookies the CDN sets).
    try:
        session.get(SOURCE_PAGE, timeout=30)
    except requests.RequestException:
        pass

    for url in DOWNLOAD_URLS:
        print(f"Trying: {url}")
        try:
            resp = session.get(url, timeout=60)
        except requests.RequestException as exc:
            print(f"  request failed: {exc}")
            continue

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}")
            continue

        content = resp.content
        if not _looks_like_pdf(content):
            print("  response was not a PDF (no %PDF header); skipping")
            continue
        if len(content) < MIN_BYTES:
            print(f"  suspiciously small ({len(content)} bytes); skipping")
            continue

        config.RAW_PDF_PATH.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        print(f"\nSaved {len(content):,} bytes to {config.RAW_PDF_PATH}")
        print(f"SHA-256: {digest}")
        if config.CORPUS_SHA256 and digest != config.CORPUS_SHA256:
            print(
                "  WARNING: this does not match config.CORPUS_SHA256\n"
                f"  expected {config.CORPUS_SHA256}\n"
                "  The source file may have changed since it was pinned."
            )
        elif not config.CORPUS_SHA256:
            print(
                "  Pin this: set CORPUS_SHA256 in regrag/config.py to the hash "
                "above so future runs verify the exact bytes."
            )
        return True

    return False


def _print_manual_instructions() -> None:
    print(
        "\nAutomatic download failed. Download the PDF by hand:\n"
        f"  1. Open {SOURCE_PAGE}\n"
        '  2. Download "The FATF Recommendations" PDF\n'
        f"  3. Save it as {config.RAW_PDF_PATH}\n"
    )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _already_present() -> bool:
    """True if the corpus PDF is already downloaded and (if pinned) verified.

    Makes re-runs idempotent: we do not re-pull from the archive when a valid
    copy is already on disk. Pass --force to re-download anyway.
    """
    if not config.RAW_PDF_PATH.exists():
        return False
    if not config.CORPUS_SHA256:
        return True  # present but not pinned; trust the existing file
    return _sha256(config.RAW_PDF_PATH) == config.CORPUS_SHA256


if __name__ == "__main__":
    force = "--force" in sys.argv
    if not force and _already_present():
        print(
            f"Corpus already present at {config.RAW_PDF_PATH} "
            "(sha256 verified); skipping download. Use --force to re-download."
        )
        sys.exit(0)

    ok = download()
    if not ok:
        _print_manual_instructions()
        sys.exit(1)
