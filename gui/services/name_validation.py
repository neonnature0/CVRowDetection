"""Validation helpers for block names and file paths derived from them."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException

_BLOCK_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_block_name_or_400(name: str) -> str:
    """Validate path-bound block names and return sanitized value.

    Allowed names: 1-64 chars, alphanumeric, underscore, hyphen.
    Must start with an alphanumeric character.
    """
    if not _BLOCK_NAME_RE.fullmatch(name):
        raise HTTPException(status_code=400, detail="Invalid block name")
    return name


def resolve_under_or_400(base_dir: Path, *parts: str) -> Path:
    """Resolve a path under base_dir and reject parent traversal attempts."""
    base = base_dir.resolve()
    resolved = base.joinpath(*parts).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid block name") from e
    return resolved
