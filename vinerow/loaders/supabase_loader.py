"""Load block definitions from a Supabase/PostGIS database.

Queries vineyard.blocks with ST_AsGeoJSON conversion via the Supabase
/pg SQL endpoint, falling back to PostgREST if unavailable.

Requires ``SUPABASE_URL`` and ``SUPABASE_SERVICE_KEY`` environment
variables (or constructor arguments).
"""

from __future__ import annotations

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)


class SupabaseLoader:
    """Load vineyard block boundaries from Supabase.

    Args:
        url: Supabase project URL.  Falls back to ``SUPABASE_URL`` env var.
        service_key: Service role key.  Falls back to ``SUPABASE_SERVICE_KEY``.
        org_id: Optional organisation account ID to filter blocks.
    """

    def __init__(
        self,
        url: str | None = None,
        service_key: str | None = None,
        org_id: str | None = None,
    ):
        self.url = url or os.environ.get("SUPABASE_URL", "")
        self.service_key = service_key or os.environ.get("SUPABASE_SERVICE_KEY", "")
        self.org_id = org_id

    def load(self) -> list[dict]:
        if not self.url or not self.service_key:
            logger.error(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env "
                "or passed to SupabaseLoader."
            )
            return []

        headers = {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

        # Build SQL
        where_clause = ""
        if self.org_id:
            safe_id = "".join(c for c in self.org_id if c in "0123456789abcdef-")
            where_clause = f"AND b.organisation_id = '{safe_id}'"

        sql = f"""
        SELECT
            b.name,
            v.name AS vineyard_name,
            ST_AsGeoJSON(b.boundary)::json AS boundary_geojson,
            b.row_spacing_m,
            b.row_orientation,
            b.row_angle,
            p.total_row_count AS row_count
        FROM vineyard.blocks b
        JOIN vineyard.vineyards v ON v.id = b.vineyard_id
        LEFT JOIN vineyard.plantings p ON p.block_id = b.id AND p.is_current = true
        WHERE b.boundary IS NOT NULL
        {where_clause}
        ORDER BY v.name, b.name
        """

        logger.info("Fetching blocks from Supabase...")
        logger.debug("SQL: %s", sql.strip())

        # Try the /pg SQL endpoint first
        try:
            resp = requests.post(
                f"{self.url}/pg",
                headers=headers,
                json={"query": sql.strip()},
                timeout=30,
            )
            if resp.status_code == 404:
                logger.warning("/pg endpoint not available, trying PostgREST")
                return self._fetch_postgrest(headers)
            resp.raise_for_status()
            rows = resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("Failed /pg endpoint: %s. Trying PostgREST.", exc)
            return self._fetch_postgrest(headers)

        if isinstance(rows, dict):
            rows = rows.get("rows", rows.get("data", []))

        blocks: list[dict] = []
        for row in rows:
            boundary = row.get("boundary_geojson")
            if boundary is None:
                continue
            if isinstance(boundary, str):
                boundary = json.loads(boundary)
            blocks.append({
                "name": row.get("name", "Unknown"),
                "vineyard_name": row.get("vineyard_name", "Unknown"),
                "boundary": boundary,
                "row_spacing_m": row.get("row_spacing_m"),
                "row_orientation": row.get("row_orientation"),
                "row_angle": row.get("row_angle"),
                "row_count": row.get("row_count"),
            })

        logger.info("Fetched %d blocks with boundaries from Supabase", len(blocks))
        return blocks

    def _fetch_postgrest(self, headers: dict) -> list[dict]:
        """Fallback via PostgREST (boundaries unavailable as GeoJSON)."""
        url = f"{self.url}/rest/v1/blocks"
        params = {
            "select": "name,row_spacing_m,row_orientation,row_angle,vineyard:vineyards(name)",
            "boundary": "not.is.null",
        }
        if self.org_id:
            params["organisation_id"] = f"eq.{self.org_id}"

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json()
        except requests.exceptions.RequestException as exc:
            logger.error("PostgREST query failed: %s", exc)
            return []

        blocks: list[dict] = []
        for row in rows:
            vi = row.get("vineyard", {})
            blocks.append({
                "name": row.get("name", "Unknown"),
                "vineyard_name": vi.get("name", "Unknown") if isinstance(vi, dict) else "Unknown",
                "boundary": None,
                "row_spacing_m": row.get("row_spacing_m"),
                "row_orientation": row.get("row_orientation"),
                "row_angle": row.get("row_angle"),
                "row_count": None,
            })

        if blocks:
            logger.warning(
                "Fetched %d blocks but boundaries are NOT available via PostgREST. "
                "Create a database function or export boundaries manually.",
                len(blocks),
            )
        return blocks
