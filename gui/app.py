"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from gui.routers import annotation, blocks, detection, tiles


def create_app() -> FastAPI:
    app = FastAPI(title="CVRowDetection", docs_url="/api/docs")

    # Routers
    app.include_router(blocks.router, prefix="/api/blocks", tags=["blocks"])
    app.include_router(detection.router, prefix="/api/detection", tags=["detection"])
    app.include_router(annotation.router, prefix="/api/annotations", tags=["annotations"])
    app.include_router(tiles.router, prefix="/api/tiles", tags=["tiles"])

    # Static files (served last so API routes take priority)
    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
