#!/usr/bin/env python3
"""Download model weights from GitHub Releases.

Usage:
    python models/download_models.py
    python models/download_models.py --token ghp_xxx   # for private repos
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = "neonnature0/CVRowDetection"
MODELS_DIR = Path(__file__).parent

# Files to download from the latest release
EXPECTED_FILES = [
    "best_model_fpn.pth",
    "mobile_sam.pt",
]


def download_release_asset(
    repo: str,
    filename: str,
    dest: Path,
    token: str | None = None,
) -> bool:
    """Download a release asset from GitHub."""
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        req = Request(api_url, headers=headers)
        with urlopen(req, timeout=30) as resp:
            import json
            release = json.loads(resp.read())
    except HTTPError as e:
        if e.code == 404:
            logger.error("No releases found for %s", repo)
        else:
            logger.error("GitHub API error: %s", e)
        return False

    # Find the asset
    asset = None
    for a in release.get("assets", []):
        if a["name"] == filename:
            asset = a
            break

    if asset is None:
        logger.warning("Asset '%s' not found in release '%s'", filename, release.get("tag_name"))
        return False

    # Download
    dl_url = asset["browser_download_url"]
    dl_headers = {}
    if token:
        dl_headers["Authorization"] = f"token {token}"

    logger.info("Downloading %s (%.1f MB)...", filename, asset["size"] / 1e6)
    try:
        req = Request(dl_url, headers=dl_headers)
        with urlopen(req, timeout=300) as resp:
            dest.write_bytes(resp.read())
        logger.info("Saved to %s", dest)
        return True
    except HTTPError as e:
        logger.error("Download failed: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token (for private repos). Also reads GITHUB_TOKEN env var.",
    )
    args = parser.parse_args()

    success = 0
    skipped = 0
    for filename in EXPECTED_FILES:
        dest = MODELS_DIR / filename
        if dest.exists():
            logger.info("Already exists: %s", dest)
            skipped += 1
            continue
        if download_release_asset(REPO, filename, dest, token=args.token):
            success += 1

    total = len(EXPECTED_FILES)
    logger.info(
        "\nDone: %d downloaded, %d already present, %d missing",
        success, skipped, total - success - skipped,
    )
    if success + skipped < total:
        logger.warning(
            "Some models are missing. Upload them to a GitHub Release "
            "or place them manually in %s", MODELS_DIR,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
