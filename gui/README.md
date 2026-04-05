# CVRowDetection GUI

Browser-based interface for the full row detection workflow: draw block boundaries, run detection, annotate rows, train ML model, and verify results.

## Setup

```bash
pip install ".[gui]"
```

## Usage

```bash
python -m gui.server
```

Opens `http://127.0.0.1:8765` in your browser.

## Local-display constraint

The annotation editor launches `annotate.py` (matplotlib) as a subprocess in a separate OS window. This requires the server to run on the same machine as the user — it will not work over a remote/headless connection.

## Sections

1. **Add Blocks** — Draw vineyard block boundaries on a MapLibre map
2. **Annotate** — Sequential workflow: detect rows, accept or edit, advance
3. **Library** — Browse all blocks, re-detect, annotate, delete
4. **Train** — Generate training data and train the ML model
5. **Verify** — Batch visual verification on random blocks
