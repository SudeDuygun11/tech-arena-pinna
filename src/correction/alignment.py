"""Local-frame (aligned) anchor patches.

Every anchor patch is expressed in the canonical ear frame, so where the rim
runs -- and therefore which way an anchor can slide -- differs from patch to
patch and the network must learn every orientation separately. Rotation
augmentation already helped (Wing + augmentation), which says orientation
matters. Here it is removed instead of averaged over.

Frame for anchor a, built only from things available at test time:
    n  surface normal at the patch centre (mean of normals within 2.5mm)
    t  contour direction at a from the PREDICTED landmarks (a-2 .. a+2 within
       its contour, one-sided at contour ends), projected onto the plane
       perpendicular to n
    b  n x t
The patch xyz and normals are rotated into (t, b, n); scalar channels
(curvature, Gaussian) are rotation-invariant and untouched; zero-padded rows stay
zero. The network's target and output live in the same frame: training target
= target @ R, world correction = output @ R.T.

Switch: set_aligned_patches(True). Off = behaviour unchanged.
"""
from __future__ import annotations

import numpy as np

from ..foundations.contours import CONTOUR_SPECS
from .patch_features import PATCH_RADIUS

HALF_SPAN = 2
NORMAL_RADIUS_MM = 2.5
_STATE = {"on": False}


def set_aligned_patches(flag: bool) -> None:
    _STATE["on"] = bool(flag)


def aligned_patches() -> bool:
    return _STATE["on"]


def _contour_range(a: int):
    for spec in CONTOUR_SPECS.values():
        lo, hi = spec["range"]
        if lo <= a < hi:
            return lo, hi
    raise ValueError(f"landmark {a} is in no contour")


def frame(raw: np.ndarray, a: int, patch: np.ndarray, radius: float | None = None) -> np.ndarray:
    """(3,3) rotation whose COLUMNS are (t, b, n) in the canonical frame."""
    radius = radius or PATCH_RADIUS
    lo, hi = _contour_range(a)
    t = raw[min(hi - 1, a + HALF_SPAN)] - raw[max(lo, a - HALF_SPAN)]
    real = np.abs(patch).sum(1) > 0
    near = real & (np.linalg.norm(patch[:, 0:3] * radius, axis=1) <= NORMAL_RADIUS_MM)
    if near.sum() < 3:
        near = real
    n = patch[near, 3:6].astype(np.float64).mean(0) if near.any() else np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(n) < 1e-6:
        n = np.array([0.0, 0.0, 1.0])
    n = n / np.linalg.norm(n)
    t = t - (t @ n) * n
    if np.linalg.norm(t) < 1e-6:                      # degenerate contour direction
        t = np.cross(n, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(n, np.array([0.0, 1.0, 0.0]))
    t = t / np.linalg.norm(t)
    return np.stack([t, np.cross(n, t), n], axis=1)


def align(patch: np.ndarray, R: np.ndarray) -> np.ndarray:
    out = patch.copy()
    real = np.abs(patch).sum(1) > 0
    out[real, 0:3] = patch[real, 0:3] @ R
    out[real, 3:6] = patch[real, 3:6] @ R
    return out.astype(patch.dtype)
