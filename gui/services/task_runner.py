"""Background task and subprocess management.

Tracks running subprocesses (e.g., annotate.py) with pid.is_alive()
checks to handle stuck process entries.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

from gui.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Track running editor subprocesses: block_name -> subprocess.Popen
_editors: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()


def launch_editor(block_name: str) -> bool:
    """Launch annotate.py for a specific block.

    Uses sys.executable to ensure the same Python/venv as the server.
    Returns True if launched, False if already running for this block.
    """
    with _lock:
        # Check if already running (with is_alive cleanup for dead processes)
        if block_name in _editors:
            proc = _editors[block_name]
            if proc.poll() is None:
                # Still running
                return False
            else:
                # Process died — clean up stale entry
                del _editors[block_name]

        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "annotate.py"), "--block", block_name],
            cwd=str(PROJECT_ROOT),
        )
        _editors[block_name] = proc
        logger.info("Launched annotate.py for %s (pid=%d)", block_name, proc.pid)
        return True


def check_editor_status(block_name: str) -> str:
    """Check if the editor subprocess is still running.

    Returns: 'running', 'exited', or 'not_started'
    """
    with _lock:
        if block_name not in _editors:
            return "not_started"
        proc = _editors[block_name]
        if proc.poll() is None:
            return "running"
        # Process has exited — clean up
        del _editors[block_name]
        return "exited"


def cleanup_editor(block_name: str):
    """Remove a finished editor from tracking."""
    with _lock:
        if block_name in _editors:
            proc = _editors[block_name]
            if proc.poll() is not None:
                del _editors[block_name]
