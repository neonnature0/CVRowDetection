"""Block registry CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gui.services import block_registry

router = APIRouter()


class CreateBlockRequest(BaseModel):
    boundary: dict  # GeoJSON Polygon


class SetDifficultyRequest(BaseModel):
    difficulty_rating: int | None  # 1–5 or null to clear


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


@router.patch("/{name}/difficulty")
def set_difficulty(name: str, req: SetDifficultyRequest):
    if req.difficulty_rating is not None and not (1 <= req.difficulty_rating <= 5):
        raise HTTPException(status_code=400, detail="difficulty_rating must be 1–5 or null")
    updated = block_registry.update_block(name, {"difficulty_rating": req.difficulty_rating})
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Block '{name}' not found")
    return updated


@router.delete("/{name}")
def delete_block(name: str):
    if not block_registry.delete_block(name):
        raise HTTPException(status_code=404, detail=f"Block '{name}' not found")
    return {"status": "deleted", "name": name}
