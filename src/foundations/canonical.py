"""Streaming access to (subject, side) examples in the canonical left-ear frame.

Every example is exposed with `landmarks` already mirrored into the left-ear
frame when side == 'right', so a single downstream model / template only ever
has to deal with one anatomical orientation.

ROLE IN THE PIPELINE
--------------------
Reading order : 4 of 20   (foundations)
Duty          : Mirrors right ears into the canonical left-ear frame.

Doubles usable data (200 subjects -> 400 ears) and lets one template and one
model serve both sides. iter_landmarks_only streams labels without loading
the large meshes.

Known issues / status:
  FIXED: mirror_y is a reflection (det = -1), so face winding must be reversed
  or every vertex normal points inward. Before the fix, right ears had 3
  normal features plus the derived curvature feature sign-flipped: 4 of 7
  inputs, on half the data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import trimesh

from .dataset import Dataset
from .geometry import mirror_y

SIDES = ("left", "right")


@dataclass
class CanonicalExample:
    subject_id: str
    side: str  # 'left' or 'right'
    landmarks: np.ndarray  # (85, 3), canonical left-ear frame


def iter_landmarks_only(dataset: Dataset, subject_ids: list[str] | None = None
                         ) -> Iterator[CanonicalExample]:
    """Yield landmarks for every (subject, side), without touching the (large) mesh files."""
    wanted = set(subject_ids) if subject_ids is not None else None
    for idx in range(len(dataset)):
        sid = dataset.get_identifier(idx)
        if wanted is not None and sid not in wanted:
            continue
        left = dataset._load_landmarks(dataset.landmarks_dir / f"{sid}_left_ear_landmarks.csv")
        right = dataset._load_landmarks(dataset.landmarks_dir / f"{sid}_right_ear_landmarks.csv")
        yield CanonicalExample(sid, "left", left)
        yield CanonicalExample(sid, "right", mirror_y(right))


def load_canonical_mesh(dataset: Dataset, subject_id: str, side: str) -> trimesh.Trimesh:
    """Full mesh (vertices + faces), mirrored into the canonical left-ear frame
    when side == 'right'.

    FACE WINDING MUST BE FLIPPED TOO. mirror_y multiplies by diag(1,-1,1), whose
    determinant is -1: a reflection, which reverses orientation. A face normal
    is cross(v1-v0, v2-v0), and for orthogonal M with det(M) = -1,

        cross(Ma, Mb) = -M @ cross(a, b)

    so the normal computed from mirrored vertices comes out as the exact
    NEGATIVE of the true mirrored normal. The vertex positions are fine; what
    goes stale is the winding, which is only an ordering of indices and so is
    untouched by mirroring -- yet it is where "which side is outside" is stored.
    Reversing each face applies a second sign flip and the two cancel.

    An earlier version of this docstring claimed winding was irrelevant "for our
    use (edge connectivity / vertex lookups only)". That was true when written
    and expired silently once patch_features started reading vertex_normals.
    Measured consequence on P0001 before this fix:

        left  ears: mean(normal . outward) = +0.4516, 85.9% outward
        right ears: mean(normal . outward) = -0.4516, 14.1% outward

    Because our curvature feature is (mean(neighbours) - v) . n, a projection
    ONTO the normal, its sign flipped too (-0.00421 left vs +0.00421 right):
    convex surface read as concave. That corrupted 3 normal features plus the
    curvature feature -- 4 of the 7 per-point inputs -- on every right ear, i.e.
    half of all training data, and inverted the curvature channel in the
    multi-view renders as well.
    """
    mesh_path = dataset.mesh_dir / f"{subject_id}.ply"
    mesh = trimesh.load(mesh_path)
    if side == "right":
        mesh.vertices = mirror_y(mesh.vertices)
        # reverse each face so orientation survives the reflection
        mesh.faces = np.asarray(mesh.faces)[:, ::-1]
    return mesh


def canonical_mesh_vertices(dataset: Dataset, subject_id: str, side: str) -> np.ndarray:
    return load_canonical_mesh(dataset, subject_id, side).vertices
