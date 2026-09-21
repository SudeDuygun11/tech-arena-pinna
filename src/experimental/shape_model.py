"""PCA-based statistical shape model (Active Shape Model style) -- an
alternative Stage-1a registration prior to the discrete k-reference
TPS/CPD ensemble.

Instead of warping k literal reference subjects onto the target, this learns
a continuous linear shape space (mean shape + principal deformation modes)
from ALL training subjects at once via Generalized Procrustes Analysis + PCA,
then fits a new target by iteratively finding surface correspondences and
solving for the shape-space coefficients that best explain them (classic
Active Shape Model fitting, Cootes et al.).

Rationale: TPS/CPD extrapolate awkwardly for a target ear that doesn't
resemble any of the k discrete references well. A PCA shape space instead
constrains predictions to the population's learned "valid shape" manifold,
which is the standard tool in the statistical-shape-model literature for
exactly this kind of anatomical generalization problem.


ROLE IN THE PIPELINE
--------------------
Reading order : not in the reading order (unused)   (experimental)
Duty          : PCA / Active Shape Model alternative to Stage 1a. NEVER WIRED IN.

A complete implementation with zero importers, never tested. Distinct from
joint_fit.py, which constrains anchors after correction; this would replace
the registration stage entirely.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..foundations.canonical import iter_landmarks_only
from ..foundations.contours import N_LANDMARKS
from ..foundations.dataset import Dataset
from ..foundations.geometry import apply_rigid, kabsch

N_COMPONENTS = 15
FIT_ITERS = 12
COEFF_CLIP_STD = 3.0
CROP_MARGIN = 18.0


@dataclass
class ShapeModel:
    mean_shape: np.ndarray  # (85, 3)
    components: np.ndarray  # (K, 255) -- rows are principal directions in flattened shape space
    stds: np.ndarray  # (K,) sqrt(eigenvalue) per component, for coefficient clipping
    crop_center: np.ndarray
    crop_radius: float


def _gpa_align_shapes(shapes: list[np.ndarray], n_iters: int = 5) -> np.ndarray:
    """Generalized Procrustes Analysis over full (85, 3) shapes (not just anchors)."""
    mean = shapes[0].copy()
    aligned = shapes
    for _ in range(n_iters):
        aligned = []
        for s in shapes:
            R, t = kabsch(s, mean)
            aligned.append(apply_rigid(s, R, t))
        mean = np.mean(aligned, axis=0)
    return mean, aligned


def build_shape_model(dataset: Dataset, subject_ids: list[str],
                       n_components: int = N_COMPONENTS,
                       crop_margin: float = CROP_MARGIN) -> ShapeModel:
    examples = list(iter_landmarks_only(dataset, subject_ids))
    shapes = [ex.landmarks for ex in examples]

    mean_shape, aligned = _gpa_align_shapes(shapes)

    flat = np.array([s.flatten() for s in aligned])  # (n_examples, 255)
    flat_mean = flat.mean(axis=0)
    centered = flat - flat_mean

    # PCA via SVD (n_examples is small, so this is cheap and numerically stable)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n_keep = min(n_components, len(S))
    components = Vt[:n_keep]  # (K, 255)
    eigenvalues = (S[:n_keep] ** 2) / max(len(shapes) - 1, 1)
    stds = np.sqrt(np.maximum(eigenvalues, 1e-12))

    centroids = np.array([ex.landmarks.mean(axis=0) for ex in examples])
    crop_center = centroids.mean(axis=0)
    extents = np.array([np.max(np.linalg.norm(ex.landmarks - crop_center, axis=1))
                         for ex in examples])
    crop_radius = extents.max() + crop_margin

    return ShapeModel(mean_shape=flat_mean.reshape(N_LANDMARKS, 3), components=components,
                       stds=stds, crop_center=crop_center, crop_radius=crop_radius)


def fit_shape_model(model: ShapeModel, target_vertices: np.ndarray,
                     n_iters: int = FIT_ITERS, coeff_clip_std: float = COEFF_CLIP_STD
                     ) -> np.ndarray:
    """Active-Shape-Model-style iterative fit: alternates (a) rigid alignment
    of the current shape estimate to the target, (b) nearest-surface
    correspondence search, (c) projecting the correspondences onto the PCA
    shape space (with coefficients clipped to a plausible range)."""
    dist = np.linalg.norm(target_vertices - model.crop_center[None, :], axis=1)
    cropped = target_vertices[dist <= model.crop_radius]
    if len(cropped) < 20:
        cropped = target_vertices[dist <= model.crop_radius * 2]
    tree = cKDTree(cropped)

    b = np.zeros(len(model.components))
    shape = model.mean_shape.copy()  # in model's own canonical (unrigid) frame

    R, t = np.eye(3), model.crop_center - model.mean_shape.mean(axis=0)
    for _ in range(n_iters):
        world_shape = apply_rigid(shape, R, t)
        _, idx = tree.query(world_shape)
        correspondences = cropped[idx]

        R, t = kabsch(shape, correspondences)
        local_target = apply_rigid(correspondences, R.T, -R.T @ t)  # correspondences in shape frame

        residual = (local_target - model.mean_shape).flatten()
        b = model.components @ residual  # project onto each principal direction
        clip = coeff_clip_std * model.stds
        b = np.clip(b, -clip, clip)

        shape = model.mean_shape + (b @ model.components).reshape(N_LANDMARKS, 3)

    return apply_rigid(shape, R, t)
