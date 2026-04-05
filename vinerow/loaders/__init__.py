"""Pluggable block-data loading backends.

Provides a unified interface for loading vineyard block definitions
from different sources: JSON files, GeoJSON files, or Supabase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class BlockLoader(Protocol):
    """Protocol for loading block boundary data from any source.

    Any object with a ``load() -> list[dict]`` method satisfies this
    protocol.  Each dict must contain at minimum:

    - ``name``: str
    - ``boundary``: GeoJSON Polygon dict with ``coordinates``

    Optional ground-truth fields:
    - ``vineyard_name``, ``row_spacing_m``, ``row_orientation``,
      ``row_angle``, ``row_count``
    """

    def load(self) -> list[dict]:
        ...


def load_blocks(source: str, **kwargs) -> list[dict]:
    """Factory: dispatch to the appropriate loader based on *source*.

    Args:
        source: One of:
            - Path to a ``.json`` file → :class:`JsonLoader`
            - Path to a ``.geojson`` file → :class:`GeoJsonLoader`
            - ``"supabase"`` → :class:`SupabaseLoader`
        **kwargs: Forwarded to the loader constructor.

    Returns:
        List of block dicts.
    """
    if source == "supabase":
        from vinerow.loaders.supabase_loader import SupabaseLoader
        return SupabaseLoader(**kwargs).load()

    path = Path(source)
    if path.suffix == ".geojson":
        from vinerow.loaders.geojson_loader import GeoJsonLoader
        return GeoJsonLoader(path).load()

    from vinerow.loaders.json_loader import JsonLoader
    return JsonLoader(path).load()
