"""Non-rigid template registration: the Stage-1 geometric prior.

A small ensemble of "reference" subjects (chosen close to the population-mean
anchor shape) is registered onto a target mesh via rigid ICP followed by a
regularized thin-plate-spline (TPS) non-rigid refinement. The resulting warp
is applied to each reference's known anchor coordinates to predict the
target's anchors; ensemble members are averaged.

Everything operates in the canonical "left ear" frame -- mirror right ears
with geometry.mirror_y before calling, and mirror predictions back after.


ROLE IN THE PIPELINE
--------------------
Reading order : 6 of 20   (registration)
Duty          : Stage 1a: warp reference landmarks onto a new mesh.

Rigid ICP (8 iterations, worst 15 percent of correspondences trimmed each
iteration) then non-rigid TPS (6 iterations, 450 farthest-point control
points, smoothing 4.0). Run per reference and averaged. CPD and a TPS/CPD
hybrid exist; TPS is the default.

Known issues / status:
  TPS_SMOOTHING = 4.0 and TRIM_FRACTION = 0.15 were never tuned. Generalised
  Cross-Validation would pick the smoothing per subject instead of one global
  constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from ..foundations.contours import ANCHOR_INDICES
from ..foundations.geometry import kabsch, apply_rigid

N_CONTROL_SURFACE_POINTS = 450
TPS_SMOOTHING = 4.0
RIGID_ICP_ITERS = 16   # was 8. Tier 1 sweep (2026-09-08): registration-only error
                        # still falling at 8 (3.5178mm) and 16 (3.3892mm, -0.1286);
                        # hadn't converged. See scripts/tier1_registration_sweep.py.
NONRIGID_ICP_ITERS = 6  # already converged in the same sweep (10 iters: +0.0002)
TRIM_FRACTION = 0.0    # was 0.15. Same sweep: monotone win from relaxing the trim,
                        # 0.15->0.0 measured -0.2165mm. TRIM_FRACTION exists to drop
                        # non-overlapping correspondences in PARTIAL-overlap ICP; our
                        # crop-vs-crop match has high overlap, so trimming was
                        # discarding good correspondences every one of 8 iterations.

# CPD (Coherent Point Drift) nonrigid registration params
CPD_BETA = 3.0          # Gaussian kernel width controlling deformation smoothness
CPD_LAMBDA = 3.0        # regularization weight
CPD_OUTLIER_WEIGHT = 0.1  # expected fraction of noise/outlier correspondences
CPD_MAX_ITERS = 50
CPD_TOL = 1e-5
CPD_TARGET_SUBSAMPLE = 4000  # cap target point count for tractable EM iterations


@dataclass
class ReferenceTemplate:
    subject_id: str
    landmarks: np.ndarray  # (85, 3) in canonical left-ear frame
    control_points: np.ndarray  # (M, 3) surface points used for registration
    crop_center: np.ndarray  # (3,)
    crop_radius: float


@dataclass
class Template:
    references: list = field(default_factory=list)  # list[ReferenceTemplate]
    global_crop_center: np.ndarray = None
    global_crop_radius: float = None


def _farthest_point_sample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(points) <= n:
        return np.arange(len(points))
    rng = np.random.default_rng(seed)
    selected = [rng.integers(len(points))]
    dist = np.linalg.norm(points - points[selected[0]], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(selected)


def build_reference(subject_id: str, landmarks: np.ndarray, mesh_vertices: np.ndarray,
                     crop_center: np.ndarray, crop_radius: float) -> ReferenceTemplate:
    dist = np.linalg.norm(mesh_vertices - crop_center[None, :], axis=1)
    cropped = mesh_vertices[dist <= crop_radius]
    idx = _farthest_point_sample(cropped, N_CONTROL_SURFACE_POINTS - len(landmarks))
    surface_sample = cropped[idx]
    # always include the real landmark points as control points (best practice for
    # landmark-transfer TPS: never make the network extrapolate near a labeled point)
    control_points = np.vstack([landmarks, surface_sample])
    return ReferenceTemplate(
        subject_id=subject_id,
        landmarks=landmarks.copy(),
        control_points=control_points,
        crop_center=crop_center.copy(),
        crop_radius=crop_radius,
    )


def _rigid_icp(source: np.ndarray, target: np.ndarray, n_iters=RIGID_ICP_ITERS,
               pinned_anchors: np.ndarray | None = None):
    """Alternating nearest-neighbour / Kabsch rigid fit.

    pinned_anchors: optional (15, 3) array of ESTIMATED positions for this
    target's 15 anchors. When given, the reference's own anchor rows (which are
    rows ANCHOR_INDICES of `source`, since control_points has the 85 landmarks
    first -- see build_reference) are paired directly with them and appended to
    every iteration's correspondence set, rather than being left to
    nearest-neighbour matching like every other control point.

    WHY (measured, 2026-09-10, registration-only on held-out ears):
        blind ICP                                3.0301mm  (non-anchor points)
        pinned to GROUND-TRUTH anchors           2.6250mm  (-0.4051)
        pinned to REAL predictions (~1.4mm err)  2.4116mm  (-0.4296, 20 ears)
    A noise sweep (0 -> 3mm jitter on the pins) stayed flat: -0.4051 at 0mm,
    -0.4025 at 1.4mm, -0.3461 at 3.0mm, never crossing back above the blind
    baseline. So the benefit comes from fixing the 6-DOF POSE, which even rough
    anchors constrain well -- it does not require accurate pins.

    Duplicated 3x because 15 anchor correspondences would otherwise be averaged
    away against ~430 surface control points in the least-squares fit.
    """
    R = np.eye(3)
    t = np.zeros(3)
    tree = cKDTree(target)
    warped = source.copy()
    anchor_src = source[list(ANCHOR_INDICES)] if pinned_anchors is not None else None
    for _ in range(n_iters):
        dist, idx = tree.query(warped)
        keep = (np.ones(len(dist), dtype=bool) if TRIM_FRACTION <= 0
                else dist <= np.quantile(dist, 1 - TRIM_FRACTION))
        src_pts, tgt_pts = source[keep], target[idx[keep]]
        if anchor_src is not None:
            src_pts = np.vstack([src_pts] + [anchor_src] * 3)
            tgt_pts = np.vstack([tgt_pts] + [pinned_anchors] * 3)
        R, t = kabsch(src_pts, tgt_pts)
        warped = apply_rigid(source, R, t)
    return R, t, warped


def _nonrigid_refine(base: np.ndarray, target: np.ndarray, n_iters=NONRIGID_ICP_ITERS,
                      smoothing=TPS_SMOOTHING):
    tree = cKDTree(target)
    warped = base.copy()
    spline = None
    for _ in range(n_iters):
        dist, idx = tree.query(warped)
        keep = dist <= np.quantile(dist, 1 - TRIM_FRACTION)
        spline = RBFInterpolator(base[keep], target[idx[keep]], kernel="thin_plate_spline",
                                  smoothing=smoothing)
        warped = spline(base)
    return spline, warped


def _cpd_nonrigid(source: np.ndarray, target: np.ndarray, beta=CPD_BETA, lam=CPD_LAMBDA,
                   w=CPD_OUTLIER_WEIGHT, max_iters=CPD_MAX_ITERS, tol=CPD_TOL):
    """Non-rigid Coherent Point Drift (Myronenko & Song, 2010).

    Unlike hard-correspondence ICP+TPS, every target point gets a soft
    (probabilistic) assignment to every source point each iteration, which is
    generally more robust when reference and target shapes differ meaningfully
    -- exactly our situation, since individual ear anatomy varies a lot and
    hard nearest-neighbor correspondence can lock onto the wrong match early
    in alignment. `source` becomes the returned warped point set; `source`
    and `target` must both be (*, 3).
    """
    M, D = source.shape
    N = target.shape[0]

    G = np.exp(-cdist(source, source, "sqeuclidean") / (2 * beta ** 2))
    Wt = np.zeros((M, D))

    sigma2 = np.sum(cdist(source, target, "sqeuclidean")) / (M * N * D)
    sigma2 = max(sigma2, 1e-6)

    target_sq_sum = np.sum(target ** 2, axis=1)

    for _ in range(max_iters):
        T = source + G @ Wt
        dist2 = cdist(T, target, "sqeuclidean")  # (M, N)

        c = (2 * np.pi * sigma2) ** (D / 2) * (w / (1 - w)) * (M / N)
        P_unnorm = np.exp(-dist2 / (2 * sigma2))
        denom = P_unnorm.sum(axis=0, keepdims=True) + c
        P = P_unnorm / np.maximum(denom, 1e-300)  # (M, N)

        P1 = P.sum(axis=1)  # (M,)
        Pt1 = P.sum(axis=0)  # (N,)
        Np = max(P1.sum(), 1e-8)
        PY = P @ target  # (M, D)

        A = G + lam * sigma2 * np.diag(1.0 / np.maximum(P1, 1e-8))
        b = (PY / np.maximum(P1, 1e-8)[:, None]) - source
        Wt_new = np.linalg.solve(A, b)

        T_new = source + G @ Wt_new
        sigma2_new = (np.sum(Pt1 * target_sq_sum) - 2 * np.trace(PY.T @ T_new)
                      + np.trace((T_new * P1[:, None]).T @ T_new)) / (Np * D)
        sigma2_new = max(sigma2_new, 1e-6)

        if abs(sigma2_new - sigma2) < tol:
            Wt, sigma2 = Wt_new, sigma2_new
            break
        Wt, sigma2 = Wt_new, sigma2_new

    return source + G @ Wt


# Per-ear reference selection (None = average every reference, the original behaviour).
_REFERENCE_SELECTION = {"k": None}


def set_reference_selection(k: int | None) -> None:
    """Average only the k references whose warped surface fits the scan best.

    Measured on fold 0 (134 held-out ears, registration only): the same 11 most
    typical references for every ear 3.298mm; best 5 of a 40-reference pool by
    this fit 2.741mm; choosing by true error (ceiling) 2.407mm. MUST be set
    identically for training-example generation and inference."""
    _REFERENCE_SELECTION["k"] = None if not k else int(k)


def reference_selection() -> int | None:
    return _REFERENCE_SELECTION["k"]


def register(reference: ReferenceTemplate, target_vertices: np.ndarray,
             crop_center: np.ndarray, crop_radius: float, method: str = "tps",
             pinned_anchors: np.ndarray | None = None, return_fit: bool = False):
    """Return predicted (85, 3) landmark positions for the target, using one reference.
    method: 'tps' (default, hard-ICP + thin-plate-spline) or 'cpd' (Coherent Point Drift,
    soft probabilistic correspondence)."""
    dist = np.linalg.norm(target_vertices - crop_center[None, :], axis=1)
    cropped = target_vertices[dist <= crop_radius]
    if len(cropped) < 20:
        # fall back to a generous global crop if the tight one missed the ear
        cropped = target_vertices[dist <= crop_radius * 2]

    R, t, rigid_source = _rigid_icp(reference.control_points, cropped,
                                     pinned_anchors=pinned_anchors)
    base_landmarks = apply_rigid(reference.landmarks, R, t)

    if method == "cpd":
        if len(cropped) > CPD_TARGET_SUBSAMPLE:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(cropped), CPD_TARGET_SUBSAMPLE, replace=False)
            cropped = cropped[sel]
        # landmarks are the first 85 rows of reference.control_points (see build_reference),
        # so they warp exactly alongside the rest of the source set -- no separate
        # extrapolation step needed.
        n_landmarks = len(reference.landmarks)
        warped_source = _cpd_nonrigid(rigid_source, cropped)
        predicted = warped_source[:n_landmarks]
        warped = warped_source
    else:
        spline, warped = _nonrigid_refine(rigid_source, cropped)
        predicted = spline(base_landmarks)
    if return_fit:
        # how closely the bent reference lies on this scan: a well-fitting
        # reference is a similar ear, so its landmarks transfer better
        return predicted, float(cKDTree(cropped).query(warped)[0].mean())
    return predicted


def register_ensemble(template: Template, target_vertices: np.ndarray, method: str = "tps",
                       pinned_anchors: np.ndarray | None = None) -> np.ndarray:
    k = _REFERENCE_SELECTION["k"]
    if k is None or k >= len(template.references):
        preds = []
        for ref in template.references:
            preds.append(
                register(ref, target_vertices, template.global_crop_center, template.global_crop_radius,
                          method=method, pinned_anchors=pinned_anchors)
            )
        return np.mean(preds, axis=0)
    preds, fits = [], []
    for ref in template.references:
        p, f = register(ref, target_vertices, template.global_crop_center, template.global_crop_radius,
                        method=method, pinned_anchors=pinned_anchors, return_fit=True)
        preds.append(p); fits.append(f)
    best = np.argsort(fits, kind="stable")[:k]
    return np.mean(np.asarray(preds)[best], axis=0)


def compute_raw_full(template: Template, target_vertices: np.ndarray, method: str = "tps",
                      pinned_anchors: np.ndarray | None = None) -> np.ndarray:
    """Single dispatch point for the three registration variants: 'tps' (default),
    'cpd', or 'hybrid' (TPS for outer_helix, CPD for the other 3 contours).

    pinned_anchors: see _rigid_icp. Used by pipeline.predict_canonical's second
    registration pass. The 'hybrid' variant does not support pinning (it takes a
    different code path) and silently ignores it -- hybrid is not the default and
    was never part of any reported result."""
    cache_file = _registration_cache_file(template, target_vertices, method, pinned_anchors)
    if cache_file is not None and cache_file.exists():
        return np.load(cache_file)
    if method == "hybrid":
        out = register_ensemble_hybrid(template, target_vertices)
    else:
        out = register_ensemble(template, target_vertices, method=method,
                                pinned_anchors=pinned_anchors)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp.npy")
        np.save(tmp, out); tmp.replace(cache_file)
    return out


def _registration_cache_file(template, target_vertices, method, pinned_anchors):
    """Opt-in exact cache (env REGISTRATION_CACHE_DIR). Registration is
    deterministic (fixed seeds), so the result is a pure function of the
    template references, crop, target vertices, method and pinned anchors --
    all hashed into the key. Training screens re-register the same held-out ears
    against the same fold template in every run; this skips that repeat work."""
    import hashlib
    import os
    from pathlib import Path
    root = os.environ.get("REGISTRATION_CACHE_DIR")
    if not root:
        return None
    h = hashlib.sha1()
    # this module's own source: any change to registration code invalidates the cache
    h.update(Path(__file__).read_bytes())
    h.update(method.encode())
    h.update(repr(("select", _REFERENCE_SELECTION["k"])).encode())
    h.update(np.ascontiguousarray(target_vertices, dtype=np.float64).tobytes())
    h.update(np.asarray(template.global_crop_center, dtype=np.float64).tobytes())
    h.update(np.float64(template.global_crop_radius).tobytes())
    for ref in template.references:
        for v in vars(ref).values():
            if isinstance(v, np.ndarray):
                h.update(np.ascontiguousarray(v).tobytes())
            else:
                h.update(repr(v).encode())
    if pinned_anchors is not None:
        h.update(np.ascontiguousarray(pinned_anchors, dtype=np.float64).tobytes())
    return Path(root) / f"{h.hexdigest()}.npy"


def register_ensemble_hybrid(template: Template, target_vertices: np.ndarray,
                              cpd_contours: tuple = ("concha_outline", "inner_helix",
                                                      "superior_antihelix")) -> np.ndarray:
    """Splice per-contour: use CPD's output for `cpd_contours`, TPS for the rest
    (default: TPS for outer_helix, CPD for the other 3 -- the split that looked
    best in the isolated comparison test). NOTE: this computes BOTH full
    ensembles and only keeps part of each -- it does not save any of CPD's
    cost, since a registration pass produces all 85 points at once."""
    from ..foundations.contours import CONTOUR_SPECS

    pred_tps = register_ensemble(template, target_vertices, method="tps")
    pred_cpd = register_ensemble(template, target_vertices, method="cpd")

    out = pred_tps.copy()
    for name in cpd_contours:
        s, e = CONTOUR_SPECS[name]["range"]
        out[s:e] = pred_cpd[s:e]
    return out


def predict_anchors_ensemble(template: Template, target_vertices: np.ndarray) -> np.ndarray:
    """Convenience: full-85 ensemble registration, sliced down to the 15 anchors."""
    full = register_ensemble(template, target_vertices)
    return full[ANCHOR_INDICES]
