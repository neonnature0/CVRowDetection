#!/usr/bin/env python3
"""
Interactive annotation tool for vineyard row positions.

Opens a matplotlib window showing the block aerial image with row
polylines. Rows use a control-point model: each row has 2 boundary
endpoints (at the mask edges) plus optional intermediate points for
curved rows. Lines between consecutive control points are straight.

Usage:
    python annotate.py                    # Open first pending block
    python annotate.py --block "B10"      # Open specific block
    python annotate.py --vineyard "Brooklands" --block "Block C"

Controls:
    Mouse:
        Left-click near row:     Select row / drag control point
        Left-click empty space:  Deselect (or pan if right-button)
        Right-drag:              Pan image
        Scroll:                  Zoom in/out (centered on cursor)
        Right-click on line:     Insert intermediate control point
        Middle-click ctrl pt:    Remove intermediate control point

    Keyboard:
        S:            Save annotation
        Z:            Undo
        Y:            Redo
        A:            Switch to Add mode (click to add straight row)
        D:            Switch to Delete mode (click row to delete)
        I:            Insert control point on selected row at cursor
        Delete:       Remove nearest control point on selected row
        Escape:       Back to Select mode / deselect
        N:            Next block
        P:            Previous block
        M:            Mark block as complete
        Q:            Quit (auto-saves)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_DIR = Path("dataset")
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
IMAGES_DIR = DATASET_DIR / "images"

# Colors
COLOR_ROW = "#ff00ff"         # magenta — default row
COLOR_SELECTED = "#00ffff"    # cyan — selected row
COLOR_MANUAL = "#ffaa00"      # orange for manually added rows
COLOR_MODIFIED = "#ff8800"    # orange for modified pipeline rows
COLOR_ENDPOINT = "white"      # endpoint control point fill
COLOR_MIDPOINT = "#ffdd00"    # intermediate control point fill
COLOR_CP_EDGE = "black"       # control point border

SELECT_RADIUS_PX = 25         # screen-pixel distance to select a row
CP_GRAB_RADIUS_PX = 12        # screen-pixel distance to grab a control point


# ---------------------------------------------------------------------------
# Data model — control-point rows
# ---------------------------------------------------------------------------

@dataclass
class RowData:
    id: int
    centerline_px: list[tuple[float, float]]  # control points [(x,y), ...]
    origin: str = "pipeline"
    modified: bool = False
    confidence: float = 0.0


@dataclass
class Action:
    type: str  # 'move_point', 'add', 'delete', 'insert_cp', 'remove_cp'
    row_id: int
    old_centerline: list[tuple[float, float]] | None = None
    new_centerline: list[tuple[float, float]] | None = None
    old_origin: str | None = None


@dataclass
class AnnotationState:
    block_name: str
    vineyard_name: str
    file_path: Path
    image_path: Path
    mask_path: Path
    image_size: tuple[int, int]  # (w, h)
    mpp: float
    angle_deg: float
    rows: list[RowData] = field(default_factory=list)
    status: str = "pending"
    gt_spacing_m: float | None = None
    gt_row_count: int | None = None
    dirty: bool = False
    undo_stack: list[Action] = field(default_factory=list)
    redo_stack: list[Action] = field(default_factory=list)
    annotation_start_time: float = 0.0

    @property
    def next_id(self) -> int:
        return max((r.id for r in self.rows), default=-1) + 1

    def sort_rows(self):
        """Sort rows by mean perpendicular position."""
        angle_rad = math.radians(self.angle_deg)
        pdx, pdy = -math.sin(angle_rad), math.cos(angle_rad)
        cx, cy = self.image_size[0] / 2.0, self.image_size[1] / 2.0

        def mean_perp(row: RowData) -> float:
            if not row.centerline_px:
                return 0.0
            perps = [(x - cx) * pdx + (y - cy) * pdy for x, y in row.centerline_px]
            return float(np.mean(perps))

        self.rows.sort(key=mean_perp)


# ---------------------------------------------------------------------------
# Polyline simplification (Douglas-Peucker) for legacy conversion
# ---------------------------------------------------------------------------

def _douglas_peucker(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Simplify a polyline, always keeping first and last point."""
    if len(points) <= 2:
        return list(points)

    # Find point with max distance from the line (first, last)
    start = np.array(points[0])
    end = np.array(points[-1])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        return [points[0], points[-1]]

    line_unit = line_vec / line_len
    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(points) - 1):
        pt = np.array(points[i])
        proj = np.dot(pt - start, line_unit)
        proj = max(0, min(line_len, proj))
        closest = start + proj * line_unit
        dist = np.linalg.norm(pt - closest)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left = _douglas_peucker(points[:max_idx + 1], epsilon)
        right = _douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def _polyline_to_control_points(
    centerline: list[tuple[float, float]], epsilon: float = 5.0,
) -> list[tuple[float, float]]:
    """Convert a dense polyline to sparse control points via Douglas-Peucker."""
    if len(centerline) <= 2:
        return list(centerline)
    simplified = _douglas_peucker(centerline, epsilon)
    # Ensure we always have at least 2 points
    if len(simplified) < 2 and len(centerline) >= 2:
        return [centerline[0], centerline[-1]]
    return simplified


# ---------------------------------------------------------------------------
# Create a straight line clipped to mask
# ---------------------------------------------------------------------------

def _make_boundary_line(
    perp: float, angle_deg: float, cx: float, cy: float, mask: np.ndarray,
) -> list[tuple[float, float]]:
    """Create a 2-point line at given perpendicular position, clipped to mask boundary."""
    h, w = mask.shape[:2]
    angle_rad = math.radians(angle_deg)
    rdx, rdy = math.cos(angle_rad), math.sin(angle_rad)
    pdx, pdy = -math.sin(angle_rad), math.cos(angle_rad)
    lx = cx + perp * pdx
    ly = cy + perp * pdy
    diag = math.sqrt(w * w + h * h)

    # Walk along the row direction, find first and last mask-interior points
    step = 2.0
    n = int(diag / step) + 1
    first = None
    last = None
    for i in range(n):
        t = -diag / 2 + i * step
        x = lx + t * rdx
        y = ly + t * rdy
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0:
            pt = (round(x, 1), round(y, 1))
            if first is None:
                first = pt
            last = pt

    if first is None or last is None:
        return []
    if first == last:
        return []
    return [first, last]


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_annotation(path: Path, mask: np.ndarray | None = None) -> AnnotationState:
    """Load annotation from JSON. Handles polyline, control_points, and legacy perp formats."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt = data.get("ground_truth", {})
    w, h = data.get("image_size", [0, 0])
    angle_deg = data.get("angle_deg", 0.0)
    cx, cy = w / 2.0, h / 2.0

    rows = []
    for r in data.get("rows", []):
        if "control_points" in r and r["control_points"]:
            # New control-point format
            cl = [(float(p[0]), float(p[1])) for p in r["control_points"]]
        elif "centerline_px" in r and r["centerline_px"]:
            # Legacy dense polyline format — simplify to control points
            raw = [(float(p[0]), float(p[1])) for p in r["centerline_px"]]
            cl = _polyline_to_control_points(raw, epsilon=5.0)
        elif "perp_position_px" in r:
            # Very old perpendicular-position format
            if mask is not None:
                cl = _make_boundary_line(r["perp_position_px"], angle_deg, cx, cy, mask)
            else:
                angle_rad = math.radians(angle_deg)
                rdx, rdy = math.cos(angle_rad), math.sin(angle_rad)
                pdx, pdy = -math.sin(angle_rad), math.cos(angle_rad)
                lx = cx + r["perp_position_px"] * pdx
                ly = cy + r["perp_position_px"] * pdy
                diag = math.sqrt(w * w + h * h) / 2
                cl = [
                    (round(lx - diag * rdx, 1), round(ly - diag * rdy, 1)),
                    (round(lx + diag * rdx, 1), round(ly + diag * rdy, 1)),
                ]
        else:
            cl = []

        rows.append(RowData(
            id=r["id"],
            centerline_px=cl,
            origin=r.get("origin", "pipeline"),
            modified=r.get("modified", False),
            confidence=r.get("confidence", 0.0),
        ))

    return AnnotationState(
        block_name=data["block_name"],
        vineyard_name=data["vineyard_name"],
        file_path=path,
        image_path=DATASET_DIR / data["image_file"],
        mask_path=DATASET_DIR / data["mask_file"],
        image_size=(w, h),
        mpp=data.get("meters_per_pixel", 0.1),
        angle_deg=angle_deg,
        rows=rows,
        status=data.get("metadata", {}).get("status", "pending"),
        gt_spacing_m=gt.get("gt_spacing_m"),
        gt_row_count=gt.get("gt_row_count"),
        annotation_start_time=time.time(),
    )


def save_annotation(state: AnnotationState) -> None:
    """Save annotation state back to JSON using control_points format.

    Also writes centerline_px for backward compatibility with evaluate_gt.py
    and the pipeline — the centerline is interpolated from control points.
    """
    with open(state.file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["angle_deg"] = round(state.angle_deg, 2)
    saved_rows = []
    for r in state.rows:
        # Interpolate dense polyline from control points for backward compat
        dense = _interpolate_dense_polyline(r.centerline_px, step=10.0)
        saved_rows.append({
            "id": r.id,
            "control_points": [[round(x, 1), round(y, 1)] for x, y in r.centerline_px],
            "centerline_px": [[round(x, 1), round(y, 1)] for x, y in dense],
            "confidence": round(r.confidence, 4),
            "origin": r.origin,
            "modified": r.modified,
        })
    data["rows"] = saved_rows

    data["metadata"]["status"] = state.status
    data["metadata"]["modified_at"] = datetime.now(timezone.utc).isoformat()
    elapsed = time.time() - state.annotation_start_time
    prev = data["metadata"].get("annotation_time_seconds") or 0
    data["metadata"]["annotation_time_seconds"] = round(prev + elapsed, 1)

    with open(state.file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    state.dirty = False
    state.annotation_start_time = time.time()
    logger.info("Saved: %s (%d rows, status=%s)", state.file_path.name, len(state.rows), state.status)


def _interpolate_dense_polyline(
    control_points: list[tuple[float, float]], step: float = 10.0,
) -> list[tuple[float, float]]:
    """Interpolate control points into a dense polyline with given step size."""
    if len(control_points) < 2:
        return list(control_points)

    result = []
    for i in range(len(control_points) - 1):
        x0, y0 = control_points[i]
        x1, y1 = control_points[i + 1]
        seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        n_steps = max(1, int(seg_len / step))
        for j in range(n_steps):
            t = j / n_steps
            result.append((round(x0 + t * (x1 - x0), 1), round(y0 + t * (y1 - y0), 1)))
    # Always include the last point
    result.append((round(control_points[-1][0], 1), round(control_points[-1][1], 1)))
    return result


def list_annotation_files() -> list[Path]:
    """List all annotation JSON files (excluding manifest)."""
    return sorted(
        p for p in ANNOTATIONS_DIR.glob("*.json")
        if p.name != "manifest.json"
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _point_to_segment_dist(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """Distance from point (px,py) to line segment (x0,y0)-(x1,y1)."""
    dx, dy = x1 - x0, y1 - y0
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.sqrt((px - x0) ** 2 + (py - y0) ** 2)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg_len_sq))
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def nearest_row_and_distance(
    state: AnnotationState, px: float, py: float,
) -> tuple[int | None, float]:
    """Find nearest row index and min distance to click point (segment-based)."""
    if not state.rows:
        return None, float("inf")
    best_idx = None
    best_dist = float("inf")
    for i, row in enumerate(state.rows):
        pts = row.centerline_px
        if len(pts) < 2:
            for p in pts:
                d = math.sqrt((px - p[0]) ** 2 + (py - p[1]) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            continue
        for j in range(len(pts) - 1):
            d = _point_to_segment_dist(px, py, pts[j][0], pts[j][1], pts[j + 1][0], pts[j + 1][1])
            if d < best_dist:
                best_dist = d
                best_idx = i
    return best_idx, best_dist


def nearest_control_point(
    row: RowData, px: float, py: float,
) -> tuple[int, float]:
    """Find nearest control point index and distance."""
    best_idx = 0
    best_dist = float("inf")
    for i, (x, y) in enumerate(row.centerline_px):
        d = math.sqrt((px - x) ** 2 + (py - y) ** 2)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, best_dist


def _insert_point_on_polyline(
    pts: list[tuple[float, float]], px: float, py: float,
) -> tuple[list[tuple[float, float]], int]:
    """Insert a new control point on the nearest segment, return updated list and new index."""
    if len(pts) < 2:
        return pts + [(px, py)], len(pts)

    best_seg = 0
    best_dist = float("inf")
    best_t = 0.5

    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            continue
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg_len_sq))
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        dist = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_seg = i
            best_t = t

    # Insert at the projected position on the best segment
    x0, y0 = pts[best_seg]
    x1, y1 = pts[best_seg + 1]
    new_pt = (round(x0 + best_t * (x1 - x0), 1), round(y0 + best_t * (y1 - y0), 1))
    new_pts = list(pts)
    new_pts.insert(best_seg + 1, new_pt)
    return new_pts, best_seg + 1


# ---------------------------------------------------------------------------
# Annotation Tool
# ---------------------------------------------------------------------------

class AnnotationTool:
    def __init__(self, annotation_files: list[Path], start_index: int = 0):
        self.files = annotation_files
        self.current_index = start_index
        self.state: AnnotationState | None = None

        # Interaction state
        self.mode = "select"  # select, add, delete
        self.selected_row_idx: int | None = None
        self.dragging_cp: bool = False
        self.drag_cp_idx: int | None = None
        self.drag_start_xy: tuple[float, float] | None = None
        self.drag_original_centerline: list[tuple[float, float]] | None = None
        self.panning = False
        self.pan_start_screen: tuple[float, float] | None = None  # screen pixels
        self.pan_xlim: tuple[float, float] | None = None
        self.pan_ylim: tuple[float, float] | None = None

        # Drawing handles
        self.row_lines: list = []
        self.cp_artists: list = []  # control point scatter/circle artists
        self.display_image = None
        self.full_mask = None

        # View state
        self._saved_xlim = None
        self._saved_ylim = None

        # Auto-save timer
        self.last_save_time = time.time()

        self._setup_figure()
        self._load_block(self.current_index)

    def _setup_figure(self):
        """Create the matplotlib figure and connect events."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        self.fig.subplots_adjust(left=0.02, right=0.85, top=0.95, bottom=0.05)

        # Mode radio buttons
        ax_radio = self.fig.add_axes([0.87, 0.7, 0.12, 0.12])
        self.radio = RadioButtons(ax_radio, ["Select", "Add", "Delete"], active=0)
        self.radio.on_clicked(self._on_mode_change)

        # Buttons
        ax_save = self.fig.add_axes([0.87, 0.58, 0.12, 0.04])
        self.btn_save = Button(ax_save, "Save (S)")
        self.btn_save.on_clicked(lambda _: self._save())

        ax_undo = self.fig.add_axes([0.87, 0.52, 0.058, 0.04])
        self.btn_undo = Button(ax_undo, "Undo")
        self.btn_undo.on_clicked(lambda _: self._undo())

        ax_redo = self.fig.add_axes([0.932, 0.52, 0.058, 0.04])
        self.btn_redo = Button(ax_redo, "Redo")
        self.btn_redo.on_clicked(lambda _: self._redo())

        ax_prev = self.fig.add_axes([0.87, 0.44, 0.058, 0.04])
        self.btn_prev = Button(ax_prev, "< Prev")
        self.btn_prev.on_clicked(lambda _: self._prev_block())

        ax_next = self.fig.add_axes([0.932, 0.44, 0.058, 0.04])
        self.btn_next = Button(ax_next, "Next >")
        self.btn_next.on_clicked(lambda _: self._next_block())

        ax_complete = self.fig.add_axes([0.87, 0.36, 0.12, 0.04])
        self.btn_complete = Button(ax_complete, "Mark Complete (M)")
        self.btn_complete.on_clicked(lambda _: self._mark_complete())

        # Info text area
        self.ax_info = self.fig.add_axes([0.87, 0.05, 0.12, 0.28])
        self.ax_info.axis("off")

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _load_block(self, index: int):
        """Load a block and redraw everything."""
        if self.state and self.state.dirty:
            save_annotation(self.state)

        self.current_index = index % len(self.files)
        path = self.files[self.current_index]

        # Load mask first for legacy format conversion
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mask_file = DATASET_DIR / meta["mask_file"]
        self.full_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        self.state = load_annotation(path, mask=self.full_mask)

        # Load and downsample image for display
        img_bgr = cv2.imread(str(self.state.image_path))
        if img_bgr is None:
            logger.error("Could not load image: %s", self.state.image_path)
            return

        h, w = img_bgr.shape[:2]
        max_dim = 2000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        self.display_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.selected_row_idx = None
        self.dragging_cp = False
        self.state.sort_rows()

        # Compute tight view bounds from mask
        if self.full_mask is not None:
            ys_mask, xs_mask = np.where(self.full_mask > 0)
            if len(xs_mask) > 0:
                pad_x = max(20, (xs_mask.max() - xs_mask.min()) * 0.05)
                pad_y = max(20, (ys_mask.max() - ys_mask.min()) * 0.05)
                self._saved_xlim = (float(xs_mask.min() - pad_x), float(xs_mask.max() + pad_x))
                self._saved_ylim = (float(ys_mask.max() + pad_y), float(ys_mask.min() - pad_y))
            else:
                self._saved_xlim = None
                self._saved_ylim = None
        else:
            self._saved_xlim = None
            self._saved_ylim = None

        self._redraw_all()

    def _redraw_all(self):
        """Full redraw of image and all rows."""
        # Preserve current view if user has zoomed/panned
        if hasattr(self.ax, 'get_xlim') and self.ax.has_data():
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            if abs(cur_xlim[1] - cur_xlim[0]) > 2:
                self._saved_xlim = cur_xlim
                self._saved_ylim = cur_ylim
        self.ax.clear()

        if self.display_image is not None:
            w_full, h_full = self.state.image_size
            self.ax.imshow(
                self.display_image,
                extent=[0, w_full, h_full, 0],
                aspect="equal",
            )

        # Draw all row lines (control-point polylines)
        self.row_lines = []
        self.cp_artists = []
        for i, row in enumerate(self.state.rows):
            is_selected = (i == self.selected_row_idx)
            color = self._row_color(row, is_selected)
            lw = 2.5 if is_selected else 1.2
            if row.centerline_px and len(row.centerline_px) >= 2:
                xs = [p[0] for p in row.centerline_px]
                ys = [p[1] for p in row.centerline_px]
            else:
                xs, ys = [], []
            line, = self.ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.85)
            self.row_lines.append(line)

            # Draw control points only on selected row
            if is_selected and len(row.centerline_px) >= 2:
                for j, (cx, cy) in enumerate(row.centerline_px):
                    is_endpoint = (j == 0 or j == len(row.centerline_px) - 1)
                    fc = COLOR_ENDPOINT if is_endpoint else COLOR_MIDPOINT
                    sz = 40 if is_endpoint else 30
                    sc = self.ax.scatter(
                        [cx], [cy], c=fc, s=sz, zorder=5,
                        edgecolors=COLOR_CP_EDGE, linewidths=1.0,
                    )
                    self.cp_artists.append(sc)

        # Restore view bounds
        if self._saved_xlim is not None:
            self.ax.set_xlim(self._saved_xlim)
            self.ax.set_ylim(self._saved_ylim)

        self._update_title()
        self._update_info()
        self.fig.canvas.draw_idle()

    def _row_color(self, row: RowData, selected: bool) -> str:
        if selected:
            return COLOR_SELECTED
        if row.origin == "manual":
            return COLOR_MANUAL
        if row.modified:
            return COLOR_MODIFIED
        return COLOR_ROW

    def _update_title(self):
        s = self.state
        status_icon = {"pending": "o", "modified": "*", "complete": "v"}.get(s.status, "?")
        mode_str = f"[{self.mode.upper()}]"
        dirty_str = " *" if s.dirty else ""
        self.fig.suptitle(
            f"{s.vineyard_name} / {s.block_name} -- {len(s.rows)} rows -- "
            f"{status_icon} {s.status}{dirty_str}  {mode_str}  "
            f"[{self.current_index+1}/{len(self.files)}]",
            fontsize=12,
        )

    def _update_info(self):
        self.ax_info.clear()
        self.ax_info.axis("off")
        s = self.state

        lines = [
            f"Rows: {len(s.rows)}",
            f"GT rows: {s.gt_row_count or 'N/A'}",
            f"Angle: {s.angle_deg:.1f} deg",
            f"GT spacing: {s.gt_spacing_m or 'N/A'}m",
            "",
            "LClick row = select",
            "RDrag = pan",
            "Scroll = zoom",
            "Drag ctrl pt = adjust",
            "RClick line = add pt",
            "MClick pt = remove pt",
            "",
            "S=Save Z=Undo Y=Redo",
            "A=Add D=Delete Esc=Sel",
            "I=Insert pt X/Del=Rm pt",
            "N/P=Next/Prev M=Done",
            "Q=Quit",
        ]

        if self.selected_row_idx is not None and self.selected_row_idx < len(s.rows):
            r = s.rows[self.selected_row_idx]
            n_pts = len(r.centerline_px)
            lines.insert(4, "")
            lines.insert(5, f"Sel: #{r.id} ({n_pts} pts)")
            lines.insert(6, f"  {r.origin}" + (" [mod]" if r.modified else ""))

        y = 0.95
        for line in lines:
            self.ax_info.text(0.0, y, line, fontsize=7.5, fontfamily="monospace",
                             transform=self.ax_info.transAxes, verticalalignment="top")
            y -= 0.052

    # --- Helpers ---

    def _select_threshold(self) -> float:
        """Convert SELECT_RADIUS_PX screen pixels to data coordinates."""
        xlim = self.ax.get_xlim()
        ax_bbox = self.ax.get_position()
        ax_width_inches = self.fig.get_size_inches()[0] * ax_bbox.width
        data_per_pixel = abs(xlim[1] - xlim[0]) / (ax_width_inches * self.fig.dpi)
        return SELECT_RADIUS_PX * data_per_pixel

    def _cp_threshold(self) -> float:
        """Convert CP_GRAB_RADIUS_PX screen pixels to data coordinates."""
        xlim = self.ax.get_xlim()
        ax_bbox = self.ax.get_position()
        ax_width_inches = self.fig.get_size_inches()[0] * ax_bbox.width
        data_per_pixel = abs(xlim[1] - xlim[0]) / (ax_width_inches * self.fig.dpi)
        return CP_GRAB_RADIUS_PX * data_per_pixel

    def _start_pan(self, event):
        """Begin panning from a mouse event (uses screen pixels to avoid feedback loop)."""
        self.panning = True
        self.pan_start_screen = (event.x, event.y)
        self.pan_xlim = self.ax.get_xlim()
        self.pan_ylim = self.ax.get_ylim()

    # --- Event handlers ---

    def _on_mode_change(self, label: str):
        self.mode = label.lower()
        self.selected_row_idx = None
        self._update_title()
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        # Right-click: pan OR insert control point on selected row
        if event.button == 3:
            if self.selected_row_idx is not None and self.mode == "select":
                # Check if click is near the selected row — insert control point
                row = self.state.rows[self.selected_row_idx]
                row_idx, dist = nearest_row_and_distance(self.state, event.xdata, event.ydata)
                if row_idx == self.selected_row_idx and dist < self._select_threshold():
                    self._insert_control_point(event.xdata, event.ydata)
                    return
            self._start_pan(event)
            return

        # Middle-click: remove control point
        if event.button == 2:
            if self.selected_row_idx is not None:
                self._remove_nearest_control_point(event.xdata, event.ydata)
            return

        if event.button != 1:
            return

        threshold = self._select_threshold()

        if self.mode == "select":
            row_idx, dist = nearest_row_and_distance(self.state, event.xdata, event.ydata)

            if row_idx is not None and dist < threshold:
                if self.selected_row_idx == row_idx:
                    # Already selected: try to grab a control point
                    row = self.state.rows[row_idx]
                    cp_idx, cp_dist = nearest_control_point(row, event.xdata, event.ydata)
                    if cp_dist < self._cp_threshold():
                        self.dragging_cp = True
                        self.drag_cp_idx = cp_idx
                        self.drag_start_xy = (event.xdata, event.ydata)
                        self.drag_original_centerline = list(row.centerline_px)
                else:
                    # Select this row
                    self.selected_row_idx = row_idx
                    self._redraw_all()
            else:
                # Clicked on empty space — deselect
                if self.selected_row_idx is not None:
                    self.selected_row_idx = None
                    self._redraw_all()

        elif self.mode == "add":
            self._add_row(event.xdata, event.ydata)

        elif self.mode == "delete":
            row_idx, dist = nearest_row_and_distance(self.state, event.xdata, event.ydata)
            if row_idx is not None and dist < threshold:
                self._delete_row(row_idx)

    def _on_release(self, event):
        if self.panning:
            self.panning = False
            return

        if self.dragging_cp and self.selected_row_idx is not None:
            row = self.state.rows[self.selected_row_idx]
            if self.drag_original_centerline is not None:
                moved = any(
                    abs(a[0] - b[0]) > 0.5 or abs(a[1] - b[1]) > 0.5
                    for a, b in zip(self.drag_original_centerline, row.centerline_px)
                )
                if moved:
                    action = Action(
                        type="move_point", row_id=row.id,
                        old_centerline=self.drag_original_centerline,
                        new_centerline=list(row.centerline_px),
                    )
                    self.state.undo_stack.append(action)
                    self.state.redo_stack.clear()
                    row.modified = True
                    self.state.dirty = True
                    if self.state.status == "pending":
                        self.state.status = "modified"
                    self._redraw_all()

        self.dragging_cp = False
        self.drag_cp_idx = None
        self.drag_start_xy = None
        self.drag_original_centerline = None

    def _on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        # Panning — use screen pixel delta to avoid data-coord feedback loop
        if self.panning and self.pan_start_screen:
            dx_px = event.x - self.pan_start_screen[0]
            dy_px = event.y - self.pan_start_screen[1]
            xlim = self.pan_xlim
            ylim = self.pan_ylim
            ax_bbox = self.ax.get_window_extent()
            data_per_px_x = (xlim[1] - xlim[0]) / ax_bbox.width
            data_per_px_y = (ylim[1] - ylim[0]) / ax_bbox.height
            self.ax.set_xlim(xlim[0] - dx_px * data_per_px_x, xlim[1] - dx_px * data_per_px_x)
            self.ax.set_ylim(ylim[0] - dy_px * data_per_px_y, ylim[1] - dy_px * data_per_px_y)
            self.fig.canvas.draw_idle()
            return

        # Dragging a single control point — direct move, no influence radius
        if self.dragging_cp and self.selected_row_idx is not None and self.drag_start_xy is not None:
            row = self.state.rows[self.selected_row_idx]
            cp_idx = self.drag_cp_idx
            # Move only the grabbed control point
            orig = self.drag_original_centerline
            dx = event.xdata - self.drag_start_xy[0]
            dy = event.ydata - self.drag_start_xy[1]
            new_pts = list(orig)
            ox, oy = orig[cp_idx]
            new_pts[cp_idx] = (ox + dx, oy + dy)
            row.centerline_px = new_pts

            # Update just this line's data (fast — no full redraw)
            if self.selected_row_idx < len(self.row_lines):
                xs = [p[0] for p in row.centerline_px]
                ys = [p[1] for p in row.centerline_px]
                self.row_lines[self.selected_row_idx].set_data(xs, ys)
            # Update control point scatter positions
            for j, artist in enumerate(self.cp_artists):
                if j < len(row.centerline_px):
                    artist.set_offsets([row.centerline_px[j]])
            self.fig.canvas.draw_idle()

        # Auto-save check
        if time.time() - self.last_save_time > 60 and self.state and self.state.dirty:
            save_annotation(self.state)
            self.last_save_time = time.time()

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        factor = 0.8 if event.button == "up" else 1.25
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xc, yc = event.xdata, event.ydata
        self.ax.set_xlim([xc + (x - xc) * factor for x in xlim])
        self.ax.set_ylim([yc + (y - yc) * factor for y in ylim])
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "s":
            self._save()
        elif event.key == "z":
            self._undo()
        elif event.key == "y":
            self._redo()
        elif event.key == "a":
            self.radio.set_active(1)
        elif event.key == "d":
            self.radio.set_active(2)
        elif event.key == "escape":
            self.radio.set_active(0)
            self.selected_row_idx = None
            self._redraw_all()
        elif event.key == "n":
            self._next_block()
        elif event.key == "p":
            self._prev_block()
        elif event.key == "m":
            self._mark_complete()
        elif event.key == "q":
            self._on_close(None)
            plt.close(self.fig)
        elif event.key == "i":
            # Insert control point at cursor position on selected row
            if self.selected_row_idx is not None and event.xdata is not None:
                self._insert_control_point(event.xdata, event.ydata)
        elif event.key in ("delete", "x"):
            # Remove nearest control point on selected row
            if self.selected_row_idx is not None and event.xdata is not None:
                self._remove_nearest_control_point(event.xdata, event.ydata)

    def _on_close(self, event):
        if self.state and self.state.dirty:
            save_annotation(self.state)

    # --- Actions ---

    def _add_row(self, click_x: float, click_y: float):
        """Add a new 2-point straight row through the click point, clipped to mask."""
        if self.full_mask is None:
            return
        w, h = self.state.image_size
        cx, cy = w / 2.0, h / 2.0
        angle_rad = math.radians(self.state.angle_deg)
        pdx, pdy = -math.sin(angle_rad), math.cos(angle_rad)
        perp = (click_x - cx) * pdx + (click_y - cy) * pdy

        points = _make_boundary_line(perp, self.state.angle_deg, cx, cy, self.full_mask)
        if len(points) < 2:
            logger.warning("No valid boundary line at click position")
            return

        new_id = self.state.next_id
        row = RowData(id=new_id, centerline_px=points, origin="manual")
        self.state.rows.append(row)
        self.state.sort_rows()
        self.state.dirty = True
        if self.state.status == "pending":
            self.state.status = "modified"

        action = Action(type="add", row_id=new_id, new_centerline=points)
        self.state.undo_stack.append(action)
        self.state.redo_stack.clear()

        self.selected_row_idx = next(
            (i for i, r in enumerate(self.state.rows) if r.id == new_id), None
        )
        self.radio.set_active(0)
        self._redraw_all()
        logger.info("Added row #%d (2 control points)", new_id)

    def _delete_row(self, idx: int):
        row = self.state.rows[idx]
        action = Action(
            type="delete", row_id=row.id,
            old_centerline=list(row.centerline_px),
            old_origin=row.origin,
        )
        self.state.undo_stack.append(action)
        self.state.redo_stack.clear()
        self.state.rows.pop(idx)
        self.state.dirty = True
        if self.state.status == "pending":
            self.state.status = "modified"
        self.selected_row_idx = None
        self._redraw_all()
        logger.info("Deleted row #%d", row.id)

    def _insert_control_point(self, px: float, py: float):
        """Insert a new intermediate control point on the selected row."""
        if self.selected_row_idx is None:
            return
        row = self.state.rows[self.selected_row_idx]
        if len(row.centerline_px) < 2:
            return

        old_cl = list(row.centerline_px)
        new_cl, new_idx = _insert_point_on_polyline(row.centerline_px, px, py)
        row.centerline_px = new_cl
        row.modified = True
        self.state.dirty = True

        action = Action(
            type="insert_cp", row_id=row.id,
            old_centerline=old_cl,
            new_centerline=list(new_cl),
        )
        self.state.undo_stack.append(action)
        self.state.redo_stack.clear()
        self._redraw_all()
        logger.info("Inserted control point at index %d on row #%d", new_idx, row.id)

    def _remove_nearest_control_point(self, px: float, py: float):
        """Remove the nearest intermediate control point (not endpoints)."""
        if self.selected_row_idx is None:
            return
        row = self.state.rows[self.selected_row_idx]
        if len(row.centerline_px) <= 2:
            logger.info("Cannot remove endpoint control points (need at least 2)")
            return

        cp_idx, cp_dist = nearest_control_point(row, px, py)
        if cp_dist > self._cp_threshold() * 2:
            return

        # Don't remove endpoints
        if cp_idx == 0 or cp_idx == len(row.centerline_px) - 1:
            logger.info("Cannot remove endpoint control points")
            return

        old_cl = list(row.centerline_px)
        new_cl = list(row.centerline_px)
        new_cl.pop(cp_idx)
        row.centerline_px = new_cl
        row.modified = True
        self.state.dirty = True

        action = Action(
            type="remove_cp", row_id=row.id,
            old_centerline=old_cl,
            new_centerline=list(new_cl),
        )
        self.state.undo_stack.append(action)
        self.state.redo_stack.clear()
        self._redraw_all()
        logger.info("Removed control point at index %d from row #%d", cp_idx, row.id)

    def _undo(self):
        if not self.state.undo_stack:
            return
        action = self.state.undo_stack.pop()

        if action.type in ("move_point", "insert_cp", "remove_cp"):
            row = next((r for r in self.state.rows if r.id == action.row_id), None)
            if row and action.old_centerline:
                row.centerline_px = list(action.old_centerline)

        elif action.type == "add":
            self.state.rows = [r for r in self.state.rows if r.id != action.row_id]

        elif action.type == "delete":
            if action.old_centerline:
                row = RowData(
                    id=action.row_id,
                    centerline_px=list(action.old_centerline),
                    origin=action.old_origin or "pipeline",
                )
                self.state.rows.append(row)
                self.state.sort_rows()

        self.state.redo_stack.append(action)
        self.state.dirty = True
        self.selected_row_idx = None
        self._redraw_all()

    def _redo(self):
        if not self.state.redo_stack:
            return
        action = self.state.redo_stack.pop()

        if action.type in ("move_point", "insert_cp", "remove_cp"):
            row = next((r for r in self.state.rows if r.id == action.row_id), None)
            if row and action.new_centerline:
                row.centerline_px = list(action.new_centerline)

        elif action.type == "add":
            if action.new_centerline:
                row = RowData(id=action.row_id, centerline_px=list(action.new_centerline), origin="manual")
                self.state.rows.append(row)
                self.state.sort_rows()

        elif action.type == "delete":
            self.state.rows = [r for r in self.state.rows if r.id != action.row_id]

        self.state.undo_stack.append(action)
        self.state.dirty = True
        self.selected_row_idx = None
        self._redraw_all()

    def _save(self):
        if self.state:
            save_annotation(self.state)
            self.last_save_time = time.time()
            self._update_title()
            self.fig.canvas.draw_idle()

    def _next_block(self):
        self._load_block(self.current_index + 1)

    def _prev_block(self):
        self._load_block(self.current_index - 1)

    def _mark_complete(self):
        if self.state:
            self.state.status = "complete"
            self.state.dirty = True
            save_annotation(self.state)
            self._update_title()
            self.fig.canvas.draw_idle()
            logger.info("Marked %s as complete", self.state.block_name)

    def run(self):
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive row annotation tool")
    parser.add_argument("--block", type=str, help="Open specific block name")
    parser.add_argument("--vineyard", type=str, help="Vineyard name (to disambiguate)")
    args = parser.parse_args()

    files = list_annotation_files()
    if not files:
        print("No annotation files found. Run prepare_dataset.py first.")
        sys.exit(1)

    start = 0
    if args.block:
        for i, f in enumerate(files):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            name_match = data["block_name"] == args.block
            vineyard_match = args.vineyard is None or data["vineyard_name"] == args.vineyard
            if name_match and vineyard_match:
                start = i
                break
        else:
            print(f"Block '{args.block}' not found in annotation files.")
            print("Available:")
            for f in files:
                with open(f, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                print(f"  {d['vineyard_name']} / {d['block_name']} ({f.name})")
            sys.exit(1)
    else:
        for i, f in enumerate(files):
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("metadata", {}).get("status") == "pending":
                start = i
                break

    print(f"Opening annotation tool ({len(files)} blocks)...")
    tool = AnnotationTool(files, start_index=start)
    tool.run()


if __name__ == "__main__":
    main()
