"""Entry point: python -m gui.server"""

from __future__ import annotations

import webbrowser

import uvicorn

from gui.app import create_app
from gui.config import HOST, PORT

app = create_app()


def main():
    url = f"http://{HOST}:{PORT}"
    print(f"Starting CVRowDetection GUI at {url}")
    webbrowser.open(url)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
