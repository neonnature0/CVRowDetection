"""Integration tests for block deletion endpoint."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from gui.app import create_app
from gui.services import block_registry, detection_cache


def test_delete_block_removes_registry_and_disk_artifacts(monkeypatch, tmp_path):
    blocks_file = tmp_path / "test_blocks.json"
    blocks_file.write_text(
        json.dumps(
            {
                "blocks": [
                    {
                        "name": "block-a",
                        "boundary": {"type": "Polygon", "coordinates": []},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    blocks_local_file = tmp_path / "test_blocks.local.json"
    annotations_dir = tmp_path / "dataset" / "annotations"
    images_dir = tmp_path / "dataset" / "images"
    detections_dir = tmp_path / "output" / "detections"

    for path in (
        annotations_dir / "block-a.json",
        images_dir / "block-a.png",
        images_dir / "block-a_mask.png",
        detections_dir / "block-a" / "result.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")

    monkeypatch.setattr(block_registry, "BLOCKS_FILE", blocks_file)
    monkeypatch.setattr(block_registry, "BLOCKS_LOCAL_FILE", blocks_local_file)
    monkeypatch.setattr(detection_cache, "ANNOTATIONS_DIR", annotations_dir)
    monkeypatch.setattr(detection_cache, "IMAGES_DIR", images_dir)
    monkeypatch.setattr(detection_cache, "DETECTIONS_DIR", detections_dir)

    with TestClient(create_app()) as client:
        response = client.delete("/api/blocks/block-a")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "name": "block-a"}

    with open(blocks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["blocks"] == []

    assert not (annotations_dir / "block-a.json").exists()
    assert not (images_dir / "block-a.png").exists()
    assert not (images_dir / "block-a_mask.png").exists()
    assert not (detections_dir / "block-a").exists()
