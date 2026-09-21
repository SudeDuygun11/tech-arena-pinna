"""Stage 2: generate the 70 non-anchor landmarks from the 15 (corrected) anchors.

Design history / why this shape: the first version traced an independent
mesh-surface shortest path between each pair of consecutive anchors. That
works well for concha_outline / inner_helix / superior_antihelix (<1mm error
even with perfect anchors) but fails badly on outer_helix (~3.7mm) because
the graph-shortest-path between the two farthest-apart anchors cuts across
the sulcus behind the ear instead of following the helix rim.

The fix: don't re-derive the contour shape from scratch. The registration
ensemble (registration.register_ensemble) already predicts a full 85-point
shape via a smooth warp built from real reference-subject neighbor points,
so its *shape* (including the outer_helix's curve) is already close to
correct -- what's wrong is just its absolute position/offset, which is
exactly what the (better-corrected) anchors capture. So: take the raw dense
warp prediction as the base shape, and smoothly blend in the anchor
correction (linear by arc-length position between the two segment anchors),
then lightly snap points that land very close to the mesh onto it.
Validated: with ground-truth anchors this gives ~0.9-1.6mm per contour,
uniformly, vs. 0.4-3.7mm (uneven) for the graph-tracing version.


ROLE IN THE PIPELINE
--------------------
Reading order : 11 of 20   (interpolation)
Duty          : Stage 2: fills in the 70 non-anchor points.

Those 70 are never predicted. Each anchor-to-anchor segment linearly blends
its two endpoint corrections along the registration's own curve shape, then
nearby points are snapped onto the mesh. Costs only 0.042mm when given
perfect anchors.

Known issues / status:
  Snaps to the nearest VERTEX, which sits 0.230mm from ground truth on
  average, while crest_attract in this same file uses the exact surface at
  0.005mm. DEFAULT_SNAP_RADIUS = 3.0 was never tuned.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS, N_LANDMARKS, contour_of, iter_segments

DEFAULT_SNAP_RADIUS = 3.0

# --- Crest attraction -------------------------------------------------------
# Measured (24 ears, scratchpad crest_diagnostic.py): ground-truth landmarks sit
# this far from the nearest high-|curvature| vertex, vs 2.63mm for random surface
# points -- outer_helix 0.81mm, inner_helix 0.98mm, concha_outline 1.08mm,
# superior_antihelix 2.28mm.
#
# Only the first three are listed here. superior_antihelix is deliberately
# EXCLUDED: at 2.28mm its crest signal is barely better than random (ratio 0.87),
# and a basin/region detector (tested separately, since that contour tracks the
# triangular-fossa rim rather than a ridge) only got it to 2.18mm -- still worse
# than the pipeline's own ~1.58mm error, so attracting toward it would pull
# points away from the truth rather than toward it.
CREST_CONTOURS = ("outer_helix", "inner_helix", "concha_outline")
CREST_PERCENTILE = 80.0   # keep the top 20% |curvature| vertices as "crest"
CREST_ALPHA = 0.5         # damped move toward the crest, not a hard snap
CREST_MAX_DIST = 2.5      # mm; beyond this the nearest crest is probably the wrong one


def crest_attract(points: np.ndarray, local_mesh, alpha: float = CREST_ALPHA,
                   percentile: float = CREST_PERCENTILE, max_dist: float = CREST_MAX_DIST
                   ) -> np.ndarray:
    """Pull the *interpolated* (non-anchor) points of the crest-tracking contours
    partway toward the nearest high-curvature vertex.

    Rationale: the 15 anchors each get their own learned correction, but the 70
    points between them are placed purely by deterministic blending -- they have
    no per-point evidence of their own. Three of the four contours provably run
    along curvature ridges (see CREST_CONTOURS above), so the ridge gives those
    points exactly the missing local evidence.

    Deliberately a *damped* move (alpha < 1) behind a distance gate rather than
    a hard snap: the measured crest is a diffuse band a few mm wide, not a clean
    line, so snapping fully onto it would trade our error for the band's.
    """
    from ..correction.patch_features import _local_curvature

    if len(local_mesh.vertices) < 50:
        return points

    from ..foundations.geometry import nearest_surface_points

    curv = np.abs(_local_curvature(local_mesh, np.arange(len(local_mesh.vertices))))
    vmask = curv >= np.percentile(curv, percentile)
    V = np.asarray(local_mesh.vertices)
    F = np.asarray(local_mesh.faces)
    # keep a face when a majority of its corners are crest, then attract to the
    # nearest point ON THAT SURFACE rather than to the nearest crest VERTEX --
    # a landmark can lie mid-triangle, so vertex targets are systematically
    # off-surface and biased toward mesh corners.
    fmask = vmask[F].sum(axis=1) >= 2
    if fmask.sum() < 10:
        return points

    anchors = set(ANCHOR_INDICES)
    eligible = [i for i in range(N_LANDMARKS)
                if i not in anchors and contour_of(i) in CREST_CONTOURS]
    if not eligible:
        return points

    out = points.copy()
    target, dist = nearest_surface_points(V, F[fmask], points[eligible])
    within = dist <= max_dist
    sel = np.array(eligible)[within]
    if len(sel):
        out[sel] = points[sel] + alpha * (target[within] - points[sel])
    return out


# Landmarks that are CONSTRUCTED rather than observed. Index 74 is defined by
# the challenge as "the end point of the continuation of the contour line for
# another 10 points with the same neighbor distance" -- arithmetic, not anatomy.
# Measured: it is the worst landmark in the whole set (3.451mm) and drags 70-73
# with it, because a patch model is being asked to detect a feature that does
# not exist. Deriving it from the contour it terminates reaches 0.395mm from
# ground-truth neighbours.
DERIVED_TERMINALS = (74,)


def resample_uniform(points: np.ndarray) -> np.ndarray:
    """Re-place interior points at uniform arc length along the polyline,
    holding both ends fixed.

    The ground truth was BUILT this way. The organisers stated the intermediate
    landmarks are "regenerated using a redistribution algorithm" between the
    annotator's fixed points, and it shows in the data: ground-truth gap spacing
    has a coefficient of variation of 1.1% (superior antihelix) to 21.8%
    (concha), while ours was 25-34% before this existed. We produced the right
    endpoints and then spaced the points between them unevenly.

    Measured effect of enforcing it, on 400 held-out ears:
        outer_helix         1.7730 -> 1.4832
        concha_outline      1.0960 -> 1.0290
        inner_helix         1.9269 -> 1.8100
        superior_antihelix  1.4525 -> 1.3464
        OVERALL             1.5326 -> 1.3837   (-9.7%)

    Applied WITHIN each anchor-to-anchor segment, not across a whole contour:
    the anchors are where the annotator actually clicked, so they stay put.
    Uniform-across-the-whole-contour was tested and is worse on inner helix
    (+0.079) because it moves the interior anchors off their predictions.

    ORDER MATTERS, AND GETTING IT WRONG COSTS THE ENTIRE GAIN. This must run as
    the LAST step of the pipeline, after polish. Calling it inside blend_correct
    was measured and does nothing: the polish network runs afterwards and moves
    all 85 points independently, so output spacing CV came back at 32.4% (inner
    helix) and 28.9% (superior antihelix) -- indistinguishable from the 31-34%
    before the change. See pipeline.predict_canonical, which applies it at the
    end. The `equidistant` argument here defaults to False for that reason.
    """
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = seg.sum()
    if total < 1e-9 or len(points) < 3:
        return points
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, total, len(points))
    return np.stack([np.interp(want, cum, points[:, k]) for k in range(3)], axis=1)


def blend_correct(raw_full: np.ndarray, corrected_anchors: dict, mesh_vertices: np.ndarray,
                   snap_radius: float = DEFAULT_SNAP_RADIUS, crest_mesh=None,
                   equidistant: bool = False, derive_terminals: bool = False,
                   mesh_faces: np.ndarray | None = None) -> np.ndarray:
    """raw_full: (85, 3) dense prediction from registration.register_ensemble.
    corrected_anchors: {global_anchor_index: (3,) corrected xyz}.
    crest_mesh: optional cropped ear mesh; if given, apply crest attraction to
    the interpolated points of the crest-tracking contours (see crest_attract).
    mesh_faces: optional (F,3) face indices. When given, the final snap uses
    the exact triangle SURFACE (nearest_surface_points) instead of the nearest
    VERTEX. Measured on 400 ears: ground truth sits 0.0054mm from the surface
    but 0.2299mm from the nearest vertex, so vertex-snapping was quantising
    every close point to mesh resolution. Confirmed post-hoc on the current-best
    run's saved predictions: 1.3736 -> 1.3564mm (-0.0172), improving 391/400
    ears. When mesh_faces is omitted, falls back to the original vertex snap
    (kept for callers that don't have faces at hand).
    Returns the full (85, 3) landmark array."""
    out = raw_full.copy()
    for seg in iter_segments():
        idxs = seg.global_indices
        n = len(idxs)
        corr_start = corrected_anchors[seg.start_anchor] - raw_full[seg.start_anchor]

        if derive_terminals and seg.end_anchor in DERIVED_TERMINALS:
            # This segment ends on a CONSTRUCTED landmark, so its predicted
            # position is not evidence. Carry the start anchor's correction
            # along the registration prior's own path (which preserves the
            # contour's shape), then place the terminal one step further on.
            # One step is stable -- the 73->74 gap matches the 72->73 gap to
            # a ratio of 1.0046 (CV 7.9%). Ten compounding steps are not:
            # extrapolating the whole tail from anchor 64 measured 21.2mm even
            # from perfect inputs, because curvature error accumulates.
            #
            # MEASURED AND REJECTED. At 400-ear scale this made landmark 74
            # WORSE, 3.451 -> 3.913mm, and dragged 70-73 with it (2.879 ->
            # 3.256mm). The 0.395mm ceiling was computed from GROUND-TRUTH
            # neighbours 71-73; our own 72/73 are not accurate enough to step
            # from, so the derivation inherits their error and adds to it.
            # Kept behind a default-off flag as a documented negative result.
            for gi in idxs:
                out[gi] = raw_full[gi] + corr_start
            if equidistant and n > 3:
                out[idxs[0]:idxs[-1]] = resample_uniform(out[idxs[0]:idxs[-1]])
            tail = out[idxs[-3]:idxs[-1]]                 # the two points before the end
            step = np.linalg.norm(np.diff(out[idxs[0]:idxs[-1]], axis=0), axis=1).mean()
            d = tail[-1] - tail[-2]
            nd = np.linalg.norm(d)
            if nd > 1e-9:
                out[seg.end_anchor] = tail[-1] + (d / nd) * step
            continue

        corr_end = corrected_anchors[seg.end_anchor] - raw_full[seg.end_anchor]
        for local_i, gi in enumerate(idxs):
            frac = local_i / (n - 1)
            out[gi] = raw_full[gi] + (1 - frac) * corr_start + frac * corr_end

        if equidistant and n > 2:
            # anchors are where the annotator clicked; only the interior moves
            out[idxs[0]:idxs[-1] + 1] = resample_uniform(out[idxs[0]:idxs[-1] + 1])

    if crest_mesh is not None:
        out = crest_attract(out, crest_mesh)

    if mesh_faces is not None and len(mesh_faces):
        from ..foundations.geometry import nearest_surface_points
        surf, dist = nearest_surface_points(mesh_vertices, mesh_faces, out)
        close = dist < snap_radius
        out[close] = surf[close]
    else:
        tree = cKDTree(mesh_vertices)
        dist, idx = tree.query(out)
        close = dist < snap_radius
        out[close] = mesh_vertices[idx][close]
    return out
