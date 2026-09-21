"""Per-reference registration bank and reference-agreement context.

Per-ear reference selection (registration.set_reference_selection) averages the k
best-fitting references. Every reference's own prediction is thrown away after
that, but how much the chosen references AGREE at a landmark is direct evidence
of how much that landmark's position can be trusted. This module keeps them.

For one scan and one template it registers every reference (same code path and
arithmetic as registration.register, TPS only) and stores per reference
    preds (K, 85, 3)  the landmark prediction
    post  (K,)        mean nearest-scan distance of the bent reference's surface
    loc   (K, 85)     the same, restricted to surface within 8mm of each landmark
The bank is cached on disk (REGISTRATION_CACHE_DIR) so different selection rules
and context definitions can be computed later without registering again.

Context per landmark (CTX_DIM = 5, all rotation-invariant so rotation
augmentation needs no change; mm quantities pre-scaled to about 0-1):
    spread      mean distance of the k chosen predictions from their mean / 2
    spread_max  largest such distance / 3
    shift       how far selection moved the estimate away from the all-reference
                mean / 3
    mean_loc    mean local fit of the chosen references * 4   (raw values ~0.1-0.25)
    mean_post   mean global fit of the chosen references * 8  (raw values ~0.1-0.2)

Switch: set_context_features(True). Everything else is unchanged when off.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from ..foundations.geometry import apply_rigid
from . import registration as _reg

CTX_DIM = 5
RADIUS = 8.0
_STATE = {"context": False}


def set_context_features(flag: bool) -> None:
    _STATE["context"] = bool(flag)


def context_features() -> bool:
    return _STATE["context"]


def _register_one(ref, V_all, crop_center, crop_radius):
    d = np.linalg.norm(V_all - crop_center[None, :], axis=1)
    cropped = V_all[d <= crop_radius]
    if len(cropped) < 20:
        cropped = V_all[d <= crop_radius * 2]
    R, t, rigid = _reg._rigid_icp(ref.control_points, cropped)
    base = apply_rigid(ref.landmarks, R, t)
    spline, warped = _reg._nonrigid_refine(rigid, cropped)
    pred = spline(base)
    dw = cKDTree(cropped).query(warped)[0]
    n = len(ref.landmarks)
    near = cdist(pred, warped[n:]) <= RADIUS
    cnt = near.sum(1)
    loc = np.where(cnt > 0, (near * dw[n:][None]).sum(1) / np.maximum(cnt, 1), dw.mean())
    return pred, float(dw.mean()), loc


def _bank_file(template, V):
    root = os.environ.get("REGISTRATION_CACHE_DIR")
    if not root:
        return None
    h = hashlib.sha1()
    h.update(Path(_reg.__file__).read_bytes())          # registration code changes invalidate
    h.update(Path(__file__).read_bytes())
    h.update(b"bank-v1")
    h.update(np.ascontiguousarray(V, dtype=np.float64).tobytes())
    h.update(np.asarray(template.global_crop_center, dtype=np.float64).tobytes())
    h.update(np.float64(template.global_crop_radius).tobytes())
    for ref in template.references:
        for v in vars(ref).values():
            h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray) else repr(v).encode())
    return Path(root) / f"bank_{h.hexdigest()}.npz"


def register_bank(template, V):
    """(preds (K,85,3), post (K,), loc (K,85)) for every reference in the template."""
    f = _bank_file(template, V)
    if f is not None and f.exists():
        z = np.load(f)
        return z["preds"], z["post"], z["loc"]
    V = np.asarray(V)
    K = len(template.references)
    preds = np.zeros((K, 85, 3)); post = np.zeros(K); loc = np.zeros((K, 85))
    for r, ref in enumerate(template.references):
        preds[r], post[r], loc[r] = _register_one(ref, V, template.global_crop_center,
                                                  template.global_crop_radius)
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, preds=preds, post=post, loc=loc)
        tmp.replace(f)
    return preds, post, loc


def select_and_context(preds, post, loc, k):
    """Mean of the k best-fitting references (identical arithmetic to
    registration.register_ensemble's selection path) and the (85, CTX_DIM) context."""
    idx = np.argsort(post, kind="stable")[:k]
    sel = preds[idx]
    raw = np.mean(sel, axis=0)
    dev = np.linalg.norm(sel - raw[None], axis=2)                       # (k, 85)
    shift = np.linalg.norm(raw - preds.mean(0), axis=1)
    ctx = np.stack([dev.mean(0) / 2.0, dev.max(0) / 3.0, shift / 3.0,
                    loc[idx].mean(0) * 4.0, np.full(85, post[idx].mean()) * 8.0], axis=1).astype(np.float32)
    return raw, ctx


def raw_and_context(template, V):
    """Registration prior with per-ear reference selection, plus its context.
    Requires registration.set_reference_selection(k)."""
    k = _reg.reference_selection()
    if not k:
        raise RuntimeError("context features need per-ear reference selection (--select-references)")
    preds, post, loc = register_bank(template, V)
    return select_and_context(preds, post, loc, min(k, len(preds)))
