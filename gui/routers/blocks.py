"""Block registry CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gui.services import block_registry

router = APIRouter()


class CreateBlockRequest(BaseModel):
    boundary: dict  # GeoJSON Polygon


@router.get("")
def list_blocks():
    return block_registry.list_blocks()


@router.get("/{name}")
def get_block(name: str):
    block = block_registry.get_block(name)
    if block is None:
        raise HTTPException(status_code=404, detail=f"Block '{name}' not found")
    return block


@router.post("", status_code=201)
def create_block(req: CreateBlockRequest):
    return block_registry.create_block(req.boundary)


@router.delete("/{name}")
def delete_block(name: str):
    if not block_registry.delete_block(name):
        raise HTTPException(status_code=404, detail=f"Block '{name}' not found")
    return {"status": "deleted", "name": name}
