"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from gui.routers import annotation, blocks, detection, elevation, progress, tiles, training, verify


def create_app() -> FastAPI:
    app = FastAPI(title="CVRowDetection", docs_url="/api/docs")

    # Routers
    # Disable caching on API and static JS/CSS during development
    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response: Response = await call_next(request)
        path = request.url.path
        if path.startswith("/api/") or path.endswith((".js", ".css", ".html")):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.include_router(blocks.router, prefix="/api/blocks", tags=["blocks"])
    app.include_router(detection.router, prefix="/api/detection", tags=["detection"])
    app.include_router(annotation.router, prefix="/api/annotations", tags=["annotations"])
    app.include_router(training.router, prefix="/api/training", tags=["training"])
    app.include_router(elevation.router, prefix="/api/elevation", tags=["elevation"])
    app.include_router(verify.router, prefix="/api/verify", tags=["verify"])
    app.include_router(progress.router, prefix="/api/progress", tags=["progress"])
    app.include_router(tiles.router, prefix="/api/tiles", tags=["tiles"])

    # Static files (served last so API routes take priority)
    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
