"""Background task and subprocess management.

Tracks running subprocesses (editors, training, data generation)
with pid.is_alive() checks to handle stuck process entries.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading

from gui.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_editors: dict[str, subprocess.Popen] = {}
_training_proc: subprocess.Popen | None = None
_generate_proc: subprocess.Popen | None = None
_lock = threading.Lock()


# ── Editor subprocess (annotate.py) ──

def launch_editor(block_name: str) -> bool:
    """Launch annotate.py for a specific block. Returns True if launched."""
    with _lock:
        if block_name in _editors:
            proc = _editors[block_name]
            if proc.poll() is None:
                return False
            del _editors[block_name]

        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "annotate.py"), "--block", block_name],
            cwd=str(PROJECT_ROOT),
        )
        _editors[block_name] = proc
        logger.info("Launched annotate.py for %s (pid=%d)", block_name, proc.pid)
        return True


def check_editor_status(block_name: str) -> str:
    """Returns: 'running', 'exited', or 'not_started'."""
    with _lock:
        if block_name not in _editors:
            return "not_started"
        proc = _editors[block_name]
        if proc.poll() is None:
            return "running"
        del _editors[block_name]
        return "exited"


# ── Training subprocess ──

def launch_training(args: list[str] | None = None) -> bool:
    """Launch training/train.py. Returns True if launched, False if already running."""
    global _training_proc
    with _lock:
        if _training_proc is not None and _training_proc.poll() is None:
            return False

        cmd = [sys.executable, "-m", "training.train"]
        if args:
            cmd.extend(args)
        _training_proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
        logger.info("Launched training (pid=%d)", _training_proc.pid)
        return True


def check_training_status() -> str:
    """Returns: 'running', 'exited', or 'not_started'."""
    global _training_proc
    with _lock:
        if _training_proc is None:
            return "not_started"
        if _training_proc.poll() is None:
            return "running"
        _training_proc = None
        return "exited"


def stop_training() -> bool:
    """Kill the training subprocess. Returns True if it was running."""
    global _training_proc
    with _lock:
        if _training_proc is None or _training_proc.poll() is not None:
            _training_proc = None
            return False
        _training_proc.terminate()
        logger.info("Terminated training process")
        _training_proc = None
        return True


# ── Generate training data subprocess ──

def launch_generate(args: list[str] | None = None) -> bool:
    """Launch generate_training_data.py. Returns True if launched."""
    global _generate_proc
    with _lock:
        if _generate_proc is not None and _generate_proc.poll() is None:
            return False

        cmd = [sys.executable, str(PROJECT_ROOT / "generate_training_data.py")]
        if args:
            cmd.extend(args)
        _generate_proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
        logger.info("Launched generate_training_data (pid=%d)", _generate_proc.pid)
        return True


def check_generate_status() -> str:
    """Returns: 'running', 'exited', or 'not_started'."""
    global _generate_proc
    with _lock:
        if _generate_proc is None:
            return "not_started"
        if _generate_proc.poll() is None:
            return "running"
        _generate_proc = None
        return "exited"


def wait_generate(timeout: float = 300) -> int:
    """Wait for generate process to finish. Returns exit code."""
    global _generate_proc
    with _lock:
        proc = _generate_proc
    if proc is None:
        return -1
    try:
        return proc.wait(timeout=timeout)
    finally:
        with _lock:
            _generate_proc = None
