"""Build a Template (population-representative reference ensemble) from a
set of training subjects, using Generalized-Procrustes-style medoid selection.


ROLE IN THE PIPELINE
--------------------
Reading order : 7 of 20   (registration)
Duty          : Builds the k-reference template from training subjects.

GPA mean anchor shape (5 iterations), then keeps the k ears with the
SMALLEST Procrustes residual: medoids, the most typical shapes, not
outliers. Also fixes the global crop geometry (landmark extent + 18mm
margin).
"""
from __future__ import annotations

import numpy as np

from ..foundations.canonical import CanonicalExample, canonical_mesh_vertices, iter_landmarks_only
from ..foundations.contours import ANCHOR_INDICES
from ..foundations.dataset import Dataset
from ..foundations.geometry import kabsch, apply_rigid
from .registration import Template, build_reference

DEFAULT_K_REFERENCES = 5
GPA_ITERS = 5
CROP_MARGIN = 18.0  # extra surface context beyond the landmark extent, in mesh units


def _gpa_mean_anchor_shape(anchor_sets: list[np.ndarray], n_iters=GPA_ITERS) -> np.ndarray:
    mean = anchor_sets[0].copy()
    for _ in range(n_iters):
        aligned = []
        for a in anchor_sets:
            R, t = kabsch(a, mean)
            aligned.append(apply_rigid(a, R, t))
        mean = np.mean(aligned, axis=0)
    return mean


def build_template(dataset: Dataset, subject_ids: list[str],
                    k_references: int = DEFAULT_K_REFERENCES,
                    crop_margin: float = CROP_MARGIN) -> Template:
    examples: list[CanonicalExample] = list(iter_landmarks_only(dataset, subject_ids))

    anchor_sets = [ex.landmarks[ANCHOR_INDICES] for ex in examples]
    mean_shape = _gpa_mean_anchor_shape(anchor_sets)

    residuals = []
    for a in anchor_sets:
        R, t = kabsch(a, mean_shape)
        aligned = apply_rigid(a, R, t)
        residuals.append(np.mean(np.linalg.norm(aligned - mean_shape, axis=1)))
    residuals = np.array(residuals)
    medoid_order = np.argsort(residuals)[:k_references]

    # crop geometry: centered on each example's own full-85-landmark centroid
    centroids = np.array([ex.landmarks.mean(axis=0) for ex in examples])
    global_crop_center = centroids.mean(axis=0)
    per_example_extent = np.array([
        np.max(np.linalg.norm(ex.landmarks - global_crop_center, axis=1)) for ex in examples
    ])
    global_crop_radius = per_example_extent.max() + crop_margin

    references = []
    for i in medoid_order:
        ex = examples[i]
        mesh_vertices = canonical_mesh_vertices(dataset, ex.subject_id, ex.side)
        crop_center = ex.landmarks.mean(axis=0)
        crop_radius = np.max(np.linalg.norm(ex.landmarks - crop_center, axis=1)) + crop_margin
        ref = build_reference(f"{ex.subject_id}_{ex.side}", ex.landmarks, mesh_vertices,
                               crop_center, crop_radius)
        references.append(ref)

    return Template(references=references, global_crop_center=global_crop_center,
                     global_crop_radius=global_crop_radius)


def save_template(template: Template, path: str):
    data = {
        "global_crop_center": template.global_crop_center,
        "global_crop_radius": np.array([template.global_crop_radius]),
        "n_references": np.array([len(template.references)]),
    }
    for i, ref in enumerate(template.references):
        data[f"ref{i}_subject_id"] = np.array([ref.subject_id])
        data[f"ref{i}_landmarks"] = ref.landmarks
        data[f"ref{i}_control_points"] = ref.control_points
        data[f"ref{i}_crop_center"] = ref.crop_center
        data[f"ref{i}_crop_radius"] = np.array([ref.crop_radius])
    np.savez_compressed(path, **data)


def load_template(path: str) -> Template:
    from .registration import ReferenceTemplate

    data = np.load(path, allow_pickle=False)
    n_refs = int(data["n_references"][0])
    references = []
    for i in range(n_refs):
        references.append(ReferenceTemplate(
            subject_id=str(data[f"ref{i}_subject_id"][0]),
            landmarks=data[f"ref{i}_landmarks"],
            control_points=data[f"ref{i}_control_points"],
            crop_center=data[f"ref{i}_crop_center"],
            crop_radius=float(data[f"ref{i}_crop_radius"][0]),
        ))
    return Template(references=references,
                     global_crop_center=data["global_crop_center"],
                     global_crop_radius=float(data["global_crop_radius"][0]))
