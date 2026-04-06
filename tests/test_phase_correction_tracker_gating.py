import numpy as np
import sys
import types


def _box_blur(img: np.ndarray, ksize: tuple[int, int]) -> np.ndarray:
    kh, kw = ksize
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.empty_like(img, dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y, x] = padded[y:y + kh, x:x + kw].mean()
    return out


sys.modules["cv2"] = types.SimpleNamespace(blur=_box_blur)

from vinerow.candidates import extraction as extraction_mod
from vinerow.candidates.extraction import _check_and_correct_phase
from vinerow.config import PipelineConfig
from vinerow.tracking.assignment import _Track, _process_strip
from vinerow.types import RowCandidate


def _make_candidates() -> list[RowCandidate]:
    candidates: list[RowCandidate] = []
    perps = [10.0, 20.0, 30.0, 40.0]
    along_x_by_strip = {0: 35.0, 1: 45.0, 2: 55.0, 3: 65.0}

    for strip in range(4):
        for i, perp in enumerate(perps):
            likelihood = 0.05 if (strip == 1 and i == 0) else 0.9
            candidates.append(
                RowCandidate(
                    x=along_x_by_strip[strip],
                    y=50.0 + perp,
                    strip_index=strip,
                    perp_position=perp,
                    strength=0.95,
                    half_width_px=2.0,
                    likelihood=likelihood,
                )
            )
    return candidates

def test_phase_correction_keeps_likelihood_and_affects_gating(monkeypatch):
    config = PipelineConfig(min_candidate_likelihood_ratio=0.3, max_consecutive_skips=1)
    mask = np.full((120, 120), 255, dtype=np.uint8)
    texture = np.full((120, 120), 1.0, dtype=np.float32)
    for row in [65, 75, 85, 95]:
        texture[row, :] = 5.0
    monkeypatch.setattr(extraction_mod, "_compute_texture_map", lambda *_args, **_kwargs: texture)

    corrected, applied = _check_and_correct_phase(
        candidates=_make_candidates(),
        luminance=np.full((120, 120), 128, dtype=np.uint8),
        mask=mask,
        angle_rad=0.0,
        spacing_px=10.0,
        cx=50.0,
        cy=50.0,
        config=config,
    )

    assert applied is True

    strip1 = sorted([c for c in corrected if c.strip_index == 1], key=lambda c: c.perp_position)
    assert len(strip1) == 4
    assert strip1[0].likelihood == 0.05

    seed = sorted([c for c in corrected if c.strip_index == 0], key=lambda c: c.perp_position)[0]
    track = _Track(track_id=0, first_candidate=seed, n_strips=4)
    all_tracks = [track]

    strip1_mean_lk = sum(c.likelihood for c in strip1) / len(strip1)

    _process_strip(
        alive_tracks=[track],
        strip_cands=strip1,
        spacing_px=10.0,
        skip_penalty=1.0,
        birth_penalty=1.0,
        max_skip_strips=1,
        config=config,
        all_tracks=all_tracks,
        next_track_id=1,
        n_strips=4,
        allow_births=False,
        strip_index=1,
        strip_mean_likelihood=strip1_mean_lk,
    )

    # Track should skip (and die due to max_consecutive_skips=1) because
    # the best-position candidate has very low likelihood relative to strip mean.
    assert track.consecutive_skips == 1
    assert track.alive is False
    assert track.candidates[1] is None
