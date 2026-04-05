"""Load block definitions from GeoJSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GeoJsonLoader:
    """Load blocks from a GeoJSON file.

    Supports FeatureCollection (returns all features), single Feature,
    or raw Geometry objects.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[dict]:
        with open(self.path, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        blocks: list[dict] = []

        if geojson.get("type") == "FeatureCollection":
            for i, feature in enumerate(geojson.get("features", [])):
                blocks.append(self._feature_to_block(feature, i))
        elif geojson.get("type") == "Feature":
            blocks.append(self._feature_to_block(geojson, 0))
        else:
            # Raw geometry
            blocks.append({
                "name": self.path.stem,
                "vineyard_name": "GeoJSON",
                "boundary": geojson,
            })

        logger.info("Loaded %d block(s) from %s", len(blocks), self.path)
        return blocks

    def _feature_to_block(self, feature: dict, index: int) -> dict:
        props = feature.get("properties", {})
        return {
            "name": props.get("name", f"{self.path.stem}_{index}"),
            "vineyard_name": props.get("vineyard_name", "GeoJSON"),
            "boundary": feature["geometry"],
            **{k: props[k] for k in ("row_spacing_m", "row_orientation", "row_angle", "row_count")
               if k in props},
        }


def load_geojson_block(path: str) -> dict:
    """Convenience function matching the legacy single-block API."""
    blocks = GeoJsonLoader(path).load()
    return blocks[0] if blocks else {}
