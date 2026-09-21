"""Build (patch_features, anchor_class, target_correction) examples for the
Stage-1 correction network, given a registration Template already built from
subjects disjoint from `subject_ids` (caller's responsibility -- this is what
keeps the correction network's training data leakage-free).

ROLE IN THE PIPELINE
--------------------
Reading order : 10 of 20   (correction)
Duty          : Generates leakage-free training examples.

generate_inner_cv_examples splits training subjects into k inner folds and
builds each fold's template from the OTHER folds, so no example is ever
produced by a template that saw that ear. Augments by jittering the patch
centre 1.5mm, simulating the registration error the model has to undo.

Known issues / status:
  The correctness of every reported number rests on this inner-CV structure.
"""
from __future__ import annotations

import gc

import numpy as np

from .anchor_model import PatchCorrector, predict_correction, train_model
from ..foundations.canonical import iter_landmarks_only, load_canonical_mesh
from ..foundations.contours import (ANCHOR_INDICES, ANCHOR_POSITION, CONTOUR_ANCHOR_COUNT,
                        CONTOUR_ANCHOR_LOCAL_POSITION, CONTOUR_SPECS, N_LANDMARKS, contour_of)
from ..foundations.dataset import Dataset
from ..interpolation.interpolation import blend_correct
from .patch_features import crop_submesh, extract_multiscale_patch, extract_patch
from ..registration.registration import Template, compute_raw_full
from ..registration import refbank as _refbank
from . import alignment as _align
from ..foundations.splits import k_fold_subject_split
from ..registration.template import build_template

AUGMENT_JITTER_STD = 1.5  # mm, simulates registration imprecision beyond what's observed
AUGMENT_DROPOUT_P = 0.15


def generate_examples(dataset: Dataset, template: Template, subject_ids: list[str],
                       augment: bool = True, seed: int = 0, registration_method: str = "tps",
                       jitter_std: float = AUGMENT_JITTER_STD, multiscale: bool = False,
                       gaussian: bool = False, geodesic: bool = False,
                       patch_points: int = None, patch_radius: float = None):
    rng = np.random.default_rng(seed)
    examples = []
    for ex in iter_landmarks_only(dataset, subject_ids):
        mesh = load_canonical_mesh(dataset, ex.subject_id, ex.side)
        ctx_full = None
        if _refbank.context_features():
            raw_full, ctx_full = _refbank.raw_and_context(template, mesh.vertices)
        else:
            raw_full = compute_raw_full(template, mesh.vertices, method=registration_method)
        local_mesh = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)
        # Nothing below touches the full mesh -- only local_mesh -- so drop it and
        # its cached derived arrays now rather than waiting for the cyclic GC.
        # See the equivalent note in generate_polish_examples.
        del mesh
        gc.collect()
        for anchor_idx in ANCHOR_INDICES:
            center = raw_full[anchor_idx]
            if augment:
                center = center + rng.normal(scale=jitter_std, size=3)
            patch = (extract_multiscale_patch(local_mesh, center, rng=rng) if multiscale
                     else extract_patch(local_mesh, center, rng=rng, gaussian=gaussian,
                                        geodesic=geodesic,
                                        **({'n_points': patch_points} if patch_points else {}),
                                        **({'radius': patch_radius} if patch_radius else {})))
            if augment and rng.random() < AUGMENT_DROPOUT_P:
                mask = rng.random(len(patch)) > 0.3
                patch = patch * mask[:, None]
            target = ex.landmarks[anchor_idx] - center
            if _align.aligned_patches():
                # patch and target into the anchor's local (contour, across, normal) frame;
                # the frame comes from the registration prior only, as at test time
                R = _align.frame(raw_full, anchor_idx, patch, patch_radius)
                patch = _align.align(patch, R)
                target = target @ R
            examples.append({
                "points": patch,
                "anchor_class": ANCHOR_POSITION[anchor_idx],
                "landmark_idx": anchor_idx,
                "target": target.astype(np.float32),
                "subject_id": ex.subject_id,
                "side": ex.side,
                **({"ctx": ctx_full[anchor_idx]} if ctx_full is not None else {}),
            })
    return examples


def generate_inner_cv_examples(dataset: Dataset, subject_ids: list[str], k_inner: int = 4,
                                k_references: int = 5, augment: bool = True, seed: int = 0,
                                verbose: bool = False, registration_method: str = "tps",
                                jitter_std: float = AUGMENT_JITTER_STD, multiscale: bool = False,
                                gaussian: bool = False, crest: bool = False,
                                geodesic: bool = False, patch_points: int = None,
                                patch_radius: float = None):
    """Leakage-free training data for the correction network: subject_ids is split
    into k_inner folds, and for each fold a Template is built from the OTHER folds
    only, then used to register (and generate examples for) the held-out fold."""
    all_examples = []
    for fold_i, (train_ids, val_ids) in enumerate(k_fold_subject_split(subject_ids, k_inner, seed)):
        template = build_template(dataset, train_ids, k_references=k_references)
        fold_examples = generate_examples(dataset, template, val_ids, augment=augment,
                                           seed=seed + fold_i, registration_method=registration_method,
                                           jitter_std=jitter_std, multiscale=multiscale,
                                           gaussian=gaussian, geodesic=geodesic,
                                           patch_points=patch_points,
                                           patch_radius=patch_radius)
        all_examples.extend(fold_examples)
        if verbose:
            print(f"  inner fold {fold_i + 1}/{k_inner}: "
                  f"{len(train_ids)} template subjects -> {len(val_ids)} scored subjects "
                  f"({len(fold_examples)} examples)")
    return all_examples


def generate_polish_examples(dataset: Dataset, template: Template, anchor_model: PatchCorrector,
                              subject_ids: list[str], augment: bool = True, seed: int = 0,
                              registration_method: str = "tps", multiscale: bool = False,
                              gaussian: bool = False, crest: bool = False,
                              geodesic: bool = False, patch_points: int = None,
                              patch_radius: float = None):
    """Training data for the Stage-2 'polish' network: for every one of the 85
    landmarks (not just anchors), predict a correction on top of the full
    Stage-1+Stage-2 pipeline output (registration -> anchor correction ->
    blend interpolation), so it directly targets the pipeline's actual
    residual error distribution at inference time.

    NOTE: `template` and `anchor_model` should generally have been fit on
    subjects other than `subject_ids` for a fully clean evaluation; when
    called with the *same* subjects used to fit them (as cross_validate.py
    does, for compute-budget reasons), this is a mild, documented leakage
    for this stage's *training* data only -- the outer held-out test
    evaluation is unaffected since it never touches this data."""
    rng = np.random.default_rng(seed)
    examples = []
    for ex in iter_landmarks_only(dataset, subject_ids):
        mesh = load_canonical_mesh(dataset, ex.subject_id, ex.side)
        ctx_full = None
        if _refbank.context_features():
            raw_full, ctx_full = _refbank.raw_and_context(template, mesh.vertices)
        else:
            raw_full = compute_raw_full(template, mesh.vertices, method=registration_method)
        local_mesh = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)

        # Free the full mesh before the per-landmark loop. Only its vertex array is
        # still needed (blend_correct's surface snap); the trimesh object also holds
        # cached edges_unique / vertex_normals / face_normals, tens of MB for a
        # 836k-vertex mesh. Those caches are reclaimed only by the cyclic GC because
        # trimesh objects contain reference cycles, so over ~100 ears peak memory
        # grows until a .ply read fails. Copying the vertices keeps behaviour
        # bit-identical while dropping everything else.
        mesh_vertices = np.asarray(mesh.vertices).copy()
        mesh_faces = np.asarray(mesh.faces).copy()   # tiny (int array); needed
        # for blend_correct's surface snap, so copied alongside vertices before
        # the full mesh (and its large cached arrays) is freed below.
        del mesh
        gc.collect()

        corrected_anchors = {}
        for anchor_idx in ANCHOR_INDICES:
            base = raw_full[anchor_idx]
            patch = (extract_multiscale_patch(local_mesh, base, rng=rng) if multiscale
                     else extract_patch(local_mesh, base, rng=rng, gaussian=gaussian,
                                        geodesic=geodesic,
                                        **({'n_points': patch_points} if patch_points else {}),
                                        **({'radius': patch_radius} if patch_radius else {})))
            from .heatmap_model import predict_any_correction
            R_a = _align.frame(raw_full, anchor_idx, patch, patch_radius) if _align.aligned_patches() else None
            correction = predict_any_correction(anchor_model, patch if R_a is None else _align.align(patch, R_a),
                                                ANCHOR_POSITION[anchor_idx],
                                                None if ctx_full is None else ctx_full[anchor_idx])
            if R_a is not None:
                correction = np.asarray(correction) @ R_a.T
            corrected_anchors[anchor_idx] = base + correction

        interp_full = blend_correct(raw_full, corrected_anchors, mesh_vertices,
                                     crest_mesh=local_mesh if crest else None,
                                     mesh_faces=mesh_faces)

        for landmark_idx in range(N_LANDMARKS):
            center = interp_full[landmark_idx]
            if augment:
                center = center + rng.normal(scale=AUGMENT_JITTER_STD * 0.5, size=3)
            patch = (extract_multiscale_patch(local_mesh, center, rng=rng) if multiscale
                     else extract_patch(local_mesh, center, rng=rng, gaussian=gaussian,
                                        geodesic=geodesic,
                                        **({'n_points': patch_points} if patch_points else {}),
                                        **({'radius': patch_radius} if patch_radius else {})))
            if augment and rng.random() < AUGMENT_DROPOUT_P:
                mask = rng.random(len(patch)) > 0.3
                patch = patch * mask[:, None]
            target = ex.landmarks[landmark_idx] - center
            examples.append({
                "points": patch,
                "anchor_class": landmark_idx,
                "target": target.astype(np.float32),
                "subject_id": ex.subject_id,
                "side": ex.side,
            })
    return examples


def train_per_contour_anchor_models(examples: list[dict], n_epochs: int = 40, verbose: bool = True,
                                     capacity: str = "small", loss_type: str = "smooth_l1",
                                     weight_decay: float = 1e-4, torch_seed: int | None = None,
                                     pooling: str = "max", optimizer: str = "adam",
                                     lr: float = 1e-3, batch_size: int = 64,
                                     loss_weighting: str = "none", augment_rotate: float = 0.0,
                                     augment_scale: float = 0.0, **train_kw) -> dict:
    """Alternative to a single shared anchor-correction network: train one
    independent PatchCorrector per contour (4 total), each seeing only that
    contour's anchors. `examples` must come from generate_examples/
    generate_inner_cv_examples, which now stores the raw `landmark_idx`.

    This directly tests the hypothesis that per-contour specialization helps
    -- vs. the risk that splitting an already data-limited training set into
    4 smaller pools hurts more than the specialization gains."""
    by_contour = {name: [] for name in CONTOUR_SPECS}
    for ex in examples:
        name = contour_of(ex["landmark_idx"])
        remapped = dict(ex)
        remapped["anchor_class"] = CONTOUR_ANCHOR_LOCAL_POSITION[name][ex["landmark_idx"]]
        by_contour[name].append(remapped)

    models = {}
    for name, contour_examples in by_contour.items():
        if verbose:
            print(f"  training {name} model on {len(contour_examples)} examples "
                  f"({CONTOUR_ANCHOR_COUNT[name]} anchor classes)...")
        # capacity / loss / seed are forwarded: without them every per-contour
        # model silently trained at the default 'small' capacity, so a per-part
        # vs shared comparison measured network size, not specialisation.
        models[name] = train_model(contour_examples, n_epochs=n_epochs,
                                    n_classes=CONTOUR_ANCHOR_COUNT[name], verbose=False,
                                    capacity=capacity, loss_type=loss_type,
                                    weight_decay=weight_decay, pooling=pooling,
                                    torch_seed=torch_seed, optimizer=optimizer, lr=lr,
                                    batch_size=batch_size, loss_weighting=loss_weighting,
                                    augment_rotate=augment_rotate, augment_scale=augment_scale,
                                    **train_kw)
    return models


def predict_correction_per_contour(models: dict, mesh_local, landmark_idx: int, center):
    """Inference counterpart to train_per_contour_anchor_models: dispatches to
    the right contour's model and remaps to its local anchor-class index."""
    name = contour_of(landmark_idx)
    local_class = CONTOUR_ANCHOR_LOCAL_POSITION[name][landmark_idx]
    patch = extract_patch(mesh_local, center)
    return predict_correction(models[name], patch, local_class)


# Derived from the observed per-contour error in our best full-scale run (1.5931mm overall:
# outer_helix 1.84, concha_outline 1.16, inner_helix 1.97, superior_antihelix 1.54) --
# upweight the persistently-harder contours (inner_helix, outer_helix), downweight the
# easiest (concha_outline), roughly proportional to relative difficulty.
DEFAULT_CONTOUR_LOSS_WEIGHTS = {
    "outer_helix": 1.15,
    "concha_outline": 0.70,
    "inner_helix": 1.25,
    "superior_antihelix": 0.95,
}


def assign_contour_weights(examples: list[dict], contour_weights: dict | None = None) -> list[dict]:
    """Return a copy of `examples` with a 'loss_weight' field set per-contour,
    for train_model/train_snapshot_ensemble's weighted loss. Examples must have
    either 'landmark_idx' (anchor-stage examples) or a raw-landmark 'anchor_class'
    (polish-stage examples, where anchor_class already IS the 0-84 landmark index)."""
    weights = contour_weights or DEFAULT_CONTOUR_LOSS_WEIGHTS
    out = []
    for ex in examples:
        idx = ex.get("landmark_idx", ex["anchor_class"])
        name = contour_of(idx)
        weighted = dict(ex)
        weighted["loss_weight"] = weights[name]
        out.append(weighted)
    return out


def boost_landmark_weight(examples: list[dict], landmark_idx: int, factor: float) -> list[dict]:
    """Multiply the loss_weight of examples for one specific landmark by `factor`
    (default loss_weight is 1.0 if none was set yet, e.g. by assign_contour_weights).
    Used to test whether index 74 -- the 'derived continuation' point, structurally
    different from the other 14 anchors (see contours.py docstring) -- benefits from
    extra training emphasis, independent of/on top of contour-level weighting."""
    out = []
    for ex in examples:
        idx = ex.get("landmark_idx", ex["anchor_class"])
        weighted = dict(ex)
        base = ex.get("loss_weight", 1.0)
        weighted["loss_weight"] = base * factor if idx == landmark_idx else base
        out.append(weighted)
    return out


def boost_hard_examples(examples: list[dict], model, hard_fraction: float,
                         boost: float, verbose: bool = True) -> list[dict]:
    """Upweight the training examples the model currently predicts WORST.

    Rationale, from our own measurements rather than intuition: filtering out
    the noisiest-looking 15% of training examples made results **+7.1% worse**.
    If those examples were label noise, dropping them would have helped. They
    did not, so they carry signal -- which points the opposite way, toward
    EMPHASISING them.

    Underfitting makes this measurable. A model that memorised its training set
    would show near-zero error everywhere and rank nothing; ours is underfit at
    every scale tested, so training error still reflects genuine difficulty.

    Implemented as loss REWEIGHTING (train_model already honours the
    'loss_weight' field) rather than duplicating rows, so it composes with
    assign_contour_weights and does not change epoch length.

    Caveat to keep in view: reweighting is redistribution. More weight on hard
    cases means less on easy ones, so it only nets positive if the hard cases
    have more reducible headroom. The comparable measured effect
    (contour weighting) was -1.7%.

    hard_fraction: share of examples treated as hard (e.g. 0.25 = worst quarter)
    boost:         multiplier applied to their existing loss_weight
    """
    from .anchor_model import predict_correction, predict_correction_ensemble

    errs = []
    for ex in examples:
        cls = ex["anchor_class"]
        if isinstance(model, list):
            pred = predict_correction_ensemble(model, ex["points"], cls)
        else:
            pred = predict_correction(model, ex["points"], cls)
        errs.append(float(np.linalg.norm(pred - ex["target"])))
    errs = np.array(errs)

    cutoff = np.quantile(errs, 1.0 - hard_fraction)
    out = []
    n_hard = 0
    for ex, e in zip(examples, errs):
        w = dict(ex)
        base = ex.get("loss_weight", 1.0)
        if e >= cutoff:
            w["loss_weight"] = base * boost
            n_hard += 1
        else:
            w["loss_weight"] = base
        out.append(w)
    if verbose:
        print(f"  hard-example mining: {n_hard}/{len(examples)} boosted x{boost} "
              f"(error >= {cutoff:.3f}mm; train err mean {errs.mean():.3f} "
              f"p95 {np.percentile(errs, 95):.3f})")
    return out
