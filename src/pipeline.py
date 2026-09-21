"""Full landmark prediction pipeline for one ear, in the canonical left-ear
frame: registration prior (Stage 1a) -> learned per-anchor correction
(Stage 1b) -> blend + surface snap to fill in the remaining 70 points
(Stage 2). See registration.py and interpolation.py for the reasoning
behind each stage.


ROLE IN THE PIPELINE
--------------------
Reading order : 12 of 20   (src root)
Duty          : Orchestrates every stage for one ear.

registration, per-anchor correction, optional curve layer, optional shape
prior, blend interpolation, optional per-point polish. Read this after the
stage modules or the ordering will not mean much.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .correction.anchor_model import PatchCorrector, predict_correction, predict_correction_ensemble
from .experimental.joint_fit import ShapePrior, apply_shape_prior
from .experimental.curve_layer import apply_curve_layer
from .correction.heatmap_model import is_heatmap, predict_heatmap_correction
from .foundations.contours import ANCHOR_INDICES, ANCHOR_POSITION, N_LANDMARKS
from .interpolation.interpolation import blend_correct, resample_uniform
from .correction.patch_features import crop_submesh, extract_multiscale_patch, extract_patch
from .registration.registration import Template, compute_raw_full
from .registration import refbank as _refbank
from .correction import alignment as _align


# No training target exceeds 16.04mm (p99.9 14.09mm). A larger or non-finite
# correction is a failure (it was a curvature spike producing 150mm) and falls
# back to the prior rather than throwing a landmark off the ear.
MAX_CORRECTION_MM = 20.0
GUARD_TRIGGERS = [0]


def _guard(correction):
    c = np.asarray(correction, dtype=float)
    if not np.all(np.isfinite(c)) or np.linalg.norm(c) > MAX_CORRECTION_MM:
        GUARD_TRIGGERS[0] += 1
        return np.zeros_like(c)
    return c


def _predict_at(model, local_mesh, center, anchor_class, tta_views: int = 1, multiscale: bool = False,
                 gaussian: bool = False, geodesic: bool = False, patch_points: int = None, patch_radius: float = None,
                 ctx=None, align_ref=None):
    """One point's correction. model may be a single PatchCorrector, a list
    (ensemble -- predictions averaged), or a dict (per-contour models).

    tta_views > 1 enables test-time augmentation: extract_patch draws a fresh
    random point subsample each call (see patch_features.extract_patch), so
    simply re-extracting and re-predicting N times and averaging gives N
    independent "views" of the same local geometry -- the same
    variance-reduction-via-averaging principle as snapshot/multi-seed
    ensembling, applied at the input level instead of the model level."""
    if isinstance(model, dict):
        from .correction.training_data import predict_correction_per_contour
        if tta_views <= 1:
            return predict_correction_per_contour(model, local_mesh, anchor_class, center)
        preds = [predict_correction_per_contour(model, local_mesh, anchor_class, center)
                 for _ in range(tta_views)]
        return np.mean(preds, axis=0)

    extract = (extract_multiscale_patch if multiscale
               else (lambda m, c: extract_patch(m, c, gaussian=gaussian, geodesic=geodesic,
                                                 **({'n_points': patch_points} if patch_points else {}),
                                                 **({'radius': patch_radius} if patch_radius else {}))))

    # Heatmap models have a different forward signature (scores + residual),
    # so they are dispatched separately. See heatmap_model.is_heatmap.
    if is_heatmap(model):
        if tta_views <= 1:
            return predict_heatmap_correction(model, extract(local_mesh, center), anchor_class)
        return np.mean([predict_heatmap_correction(model, extract(local_mesh, center), anchor_class)
                        for _ in range(tta_views)], axis=0)

    def _pc(patch):
        R = None
        if align_ref is not None:
            # aligned-patch models predict in the anchor's local frame (see correction/alignment.py)
            R = _align.frame(align_ref[0], align_ref[1], patch, patch_radius)
            patch = _align.align(patch, R)
        out = predict_correction_ensemble(model, patch, anchor_class, ctx) if isinstance(model, list) \
            else predict_correction(model, patch, anchor_class, ctx)
        return out if R is None else np.asarray(out) @ R.T

    if tta_views <= 1:
        return _pc(extract(local_mesh, center))

    preds = []
    for _ in range(tta_views):
        preds.append(_pc(extract(local_mesh, center)))
    return np.mean(preds, axis=0)


def _member_subset(models, idx: int, k: int | None):
    """Assign a ROTATING subset of ensemble members to landmark `idx`.

    Averaging all members for every landmark gives each point the same blended
    bias, so neighbouring points make highly correlated errors -- and
    correlated error at 5-10mm is exactly what a shape prior cannot filter
    (see scripts/correlated_noise_test.py). Giving adjacent landmarks
    DIFFERENT predictors decorrelates them, at the cost of a noisier
    per-point estimate (ensembling was worth -5.5%).

    That trade is deliberate: the prior removes independent noise far better
    than correlated noise (1.48 -> 0.71mm vs 1.48 -> 1.13mm), so accepting a
    worse individual prediction can still lower the final error.

    NOTE the untested assumption: adjacent landmarks still read heavily
    overlapping patches (~70% at 3mm spacing with a 7mm radius), so different
    weights may make near-identical mistakes anyway. Verify with
    scripts/error_correlation.py rather than assuming this works.

    k=None or k>=len(models) reproduces the current behaviour exactly.
    """
    n = len(models)
    if k is None or k >= n or n <= 1:
        return models
    start = (idx * k) % n
    return [models[(start + j) % n] for j in range(k)]


def predict_canonical(template: Template, model: PatchCorrector | None,
                       mesh: trimesh.Trimesh, polish_model: PatchCorrector | None = None,
                       registration_method: str = "tps", tta_views: int = 1,
                       multiscale: bool = False, gaussian: bool = False,
                       crest: bool = False, shape_prior: "ShapePrior | None" = None,
                       prior_strength: float = 0.05,
                       members_per_point: int | None = None,
                       geodesic: bool = False, patch_points: int = None,
                       patch_radius: float = None, curve_model=None,
                       equidistant: bool = True,
                       derive_terminals: bool = False,
                       two_pass: bool = False,
                       n_passes: int | None = None) -> np.ndarray:
    """Predict the full (85, 3) landmark array for a mesh already in the
    canonical left-ear frame (mirror right ears before calling).

    Stage 1a: registration ensemble (macro shape prior). registration_method:
    'tps' (default), 'cpd', or 'hybrid' (TPS for outer_helix, CPD elsewhere).
    Stage 1b (`model`): learned per-anchor correction.
    Stage 2: blend the corrected anchors into the dense registration shape.
    Stage 2b (`polish_model`, optional): learned per-point refinement on top
    of the full Stage-2 output -- targets the residual error left over from
    deterministic interpolation (see training_data.generate_polish_examples).
    tta_views: >1 enables test-time augmentation (average predictions over
    multiple randomly-resampled patch views per point).
    """
    local_mesh = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)

    def _correct_anchors(raw, ctx_rows=None):
        """Stage 1b for one registration result. Factored out because two_pass
        runs it twice: once on the blind registration to get rough anchors, then
        again on the anchor-pinned re-registration."""
        out = {}
        ecpv_local = {} if isinstance(model, list) else None
        for anchor_idx in ANCHOR_INDICES:
            base = raw[anchor_idx]
            if model is not None:
                cls = ANCHOR_POSITION[anchor_idx] if not isinstance(model, dict) else anchor_idx
                if isinstance(model, list) and shape_prior is not None:
                    # need the individual members' predictions to derive E-CPV
                    patch = (extract_multiscale_patch(local_mesh, base) if multiscale
                             else extract_patch(local_mesh, base, gaussian=gaussian,
                                                 geodesic=geodesic,
                                                 **({'n_points': patch_points} if patch_points else {}),
                                                 **({'radius': patch_radius} if patch_radius else {})))
                    per = np.array([predict_correction(m, patch, cls) for m in model])
                    correction = per.mean(axis=0)
                    ecpv_local[anchor_idx] = float(np.linalg.norm(per - correction, axis=1).mean())
                else:
                    correction = _predict_at(model, local_mesh, base, cls, tta_views,
                                              multiscale, gaussian, geodesic, patch_points,
                                              patch_radius,
                                              ctx=None if ctx_rows is None else ctx_rows[anchor_idx],
                                              align_ref=(raw, anchor_idx) if _align.aligned_patches() else None)
                base = base + _guard(correction)
            out[anchor_idx] = base
        return out, ecpv_local

    ctx_rows = None
    if model is not None and _refbank.context_features():
        raw_full, ctx_rows = _refbank.raw_and_context(template, mesh.vertices)
    else:
        raw_full = compute_raw_full(template, mesh.vertices, method=registration_method)
    corrected_anchors, ecpv = _correct_anchors(raw_full, ctx_rows)

    # How many TOTAL registration passes to run. `two_pass=True` is the boolean
    # spelling of n_passes=2 and is kept so existing callers do not change;
    # n_passes wins when both are given. Pass counts above 2 exist to answer
    # whether the pin-refine loop COMPOUNDS -- see scripts/pass_iteration_test.py.
    total_passes = n_passes if n_passes is not None else (2 if two_pass else 1)

    for _ in range(max(0, total_passes - 1) if model is not None else 0):
        # ANOTHER REGISTRATION PASS, pinned to the previous pass's anchors.
        #
        # Registration originally ran blind: rigid ICP started from identity with
        # no idea where this ear's anchors were, and landmarks were pure OUTPUT.
        # Measured (registration-only, held-out ears, 2026-09-10): pinning the
        # rigid fit to estimated anchors improves the 70 NON-anchor points by
        # -0.43mm, and a 0->3mm noise sweep on the pins stayed flat -- the gain
        # comes from fixing the 6-DOF pose, which even rough anchors constrain
        # well. See registration._rigid_icp for the full numbers.
        pinned = np.array([corrected_anchors[a] for a in ANCHOR_INDICES])
        raw_full = compute_raw_full(template, mesh.vertices, method=registration_method,
                                     pinned_anchors=pinned)
        corrected_anchors, ecpv = _correct_anchors(raw_full)

    if curve_model is not None:
        # Let the 15 anchors see their curve-mates before committing. Placed
        # here -- after per-anchor correction, before interpolation -- because
        # blend_correct propagates anchor positions into the other 70 points,
        # so any anchor repair must happen first.
        corrected_anchors = apply_curve_layer(curve_model, corrected_anchors)

    if shape_prior is not None and ecpv is not None:
        # Uniform weights were measured WORSE than no prior at all, so the prior
        # is applied only when per-anchor confidence is available (i.e. an
        # ensemble). See joint_fit module docstring.
        corrected_anchors = apply_shape_prior(corrected_anchors, ecpv, shape_prior,
                                               strength=prior_strength)

    interp_full = blend_correct(raw_full, corrected_anchors, mesh.vertices,
                                 crest_mesh=local_mesh if crest else None,
                                 equidistant=equidistant,
                                 derive_terminals=derive_terminals,
                                 mesh_faces=np.asarray(mesh.faces))

    def _respace(arr):
        """Uniform arc-length spacing within each anchor segment -- the LAST
        step, so nothing downstream can undo it. The ground truth was built by
        redistributing points evenly between the annotator's fixed points;
        enforcing it measured 1.5326 -> 1.3837mm on 400 held-out ears."""
        if not equidistant:
            return arr
        from .foundations.contours import CONTOUR_SPECS
        out = arr.copy()
        for spec in CONTOUR_SPECS.values():
            anc = spec["anchors"]
            for u, v in zip(anc[:-1], anc[1:]):
                out[u:v + 1] = resample_uniform(out[u:v + 1])
        return out

    if polish_model is None:
        return _respace(interp_full)

    polished = interp_full.copy()
    for landmark_idx in range(N_LANDMARKS):
        center = interp_full[landmark_idx]
        pm = (_member_subset(polish_model, landmark_idx, members_per_point)
              if isinstance(polish_model, list) else polish_model)
        correction = _predict_at(pm, local_mesh, center, landmark_idx, tta_views,
                                  multiscale, gaussian, geodesic, patch_points, patch_radius)
        polished[landmark_idx] = center + _guard(correction)
    return _respace(polished)
