"""Stage 1 provenance tests: canonicalisation and splitting.

WHY THESE EXIST
---------------
The mirroring bug found on 2026-08-29 was not a coding error. The code did
exactly what its docstring said, and the docstring said:

    "Face winding is irrelevant for our use (edge connectivity / vertex
     lookups only), so mirroring is not un-flipped."

That was TRUE when written -- registration needed only vertex positions. It
stopped being true the moment patch_features started reading vertex_normals,
and nothing re-checked it. The result: 3 normal features plus the derived
curvature feature were sign-flipped on every right ear, i.e. 4 of 7 model
inputs on half of all training data, for an unknown number of months.

A comment cannot notice when its own assumption expires. An assertion can.
Every claim Stage 1 makes is encoded here as a test that fails loudly if the
property stops holding.

These use SYNTHETIC geometry, so they run in seconds, need no dataset, and can
run while a training job holds the real meshes.

Run:  python -m pytest tests/ -v        (or: python tests/test_stage1_canonicalisation.py)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import load_canonical_mesh
from src.foundations.dataset import Dataset
from src.foundations.geometry import mirror_y
from src.foundations.splits import k_fold_subject_split


# --------------------------------------------------------------- helpers

def _sphere(subdivisions: int = 3, radius: float = 40.0, offset=(0.0, 90.0, 0.0)):
    """A closed surface with unambiguous outward normals, offset from the origin
    the way a real ear sits away from the scan origin."""
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    m.vertices = np.asarray(m.vertices) + np.asarray(offset, dtype=float)
    return m


def _outwardness(mesh) -> float:
    """Mean cosine between each vertex normal and the outward radial direction.
    Positive means normals point out of the surface; negative means inward."""
    V = np.asarray(mesh.vertices, dtype=float)
    N = np.asarray(mesh.vertex_normals, dtype=float)
    out = V - V.mean(axis=0)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return float(np.einsum("ij,ij->i", N, out).mean())


def _fake_dataset(tmp: Path):
    """A minimal on-disk dataset holding one synthetic subject, so the REAL
    load_canonical_mesh code path is exercised rather than a reimplementation."""
    (tmp / "mesh").mkdir(parents=True, exist_ok=True)
    (tmp / "landmarks").mkdir(parents=True, exist_ok=True)
    _sphere().export(tmp / "mesh" / "P0001.ply")
    pts = _sphere(subdivisions=1).vertices[:85]
    for side in ("left", "right"):
        with open(tmp / "landmarks" / f"P0001_{side}_ear_landmarks.csv", "w") as fh:
            for i, p in enumerate(pts):
                fh.write(f"{i},[{p[0]} {p[1]} {p[2]}]\n")
    return Dataset(mesh_dir=str(tmp / "mesh"), landmarks_dir=str(tmp / "landmarks"))


# --------------------------------------------------------------- mirroring

def test_mirror_is_an_involution():
    """Mirroring twice returns the original exactly -- no drift, no resampling."""
    rng = np.random.default_rng(0)
    p = rng.normal(size=(500, 3)) * 40.0
    assert np.allclose(mirror_y(mirror_y(p)), p, atol=0.0), "mirror_y is not exactly self-inverse"


def test_mirror_preserves_distances():
    """A reflection is an isometry: shape is not distorted, only handedness."""
    rng = np.random.default_rng(1)
    p = rng.normal(size=(200, 3)) * 40.0
    d0 = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    d1 = np.linalg.norm(mirror_y(p)[:, None] - mirror_y(p)[None, :], axis=-1)
    assert np.allclose(d0, d1, atol=1e-9), "mirror_y changed inter-point distances"


def test_mirror_reverses_orientation():
    """The property that caused the bug, asserted directly.

    For an orthogonal M with det(M) = -1,  cross(Ma, Mb) = -M @ cross(a, b).
    A normal is a pseudovector: it does NOT survive a reflection unchanged.
    """
    M = np.diag([1.0, -1.0, 1.0])
    assert np.isclose(np.linalg.det(M), -1.0)
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(np.cross(M @ a, M @ b), -(M @ np.cross(a, b))), \
        "reflection algebra changed -- the winding fix is built on this identity"


def test_canonical_mesh_normals_point_outward_on_BOTH_sides():
    """THE REGRESSION TEST for the 2026-08-29 defect.

    Before the fix this measured +0.4516 for left and -0.4516 for right on real
    data: exact negatives. Both sides must now agree in sign.
    """
    with tempfile.TemporaryDirectory() as td:
        ds = _fake_dataset(Path(td))
        left = _outwardness(load_canonical_mesh(ds, "P0001", "left"))
        right = _outwardness(load_canonical_mesh(ds, "P0001", "right"))
    assert left > 0.5, f"left-ear normals are not outward ({left:.4f})"
    assert right > 0.5, (
        f"RIGHT-ear normals point inward ({right:.4f}). Face winding is not being "
        "reversed after mirror_y. This silently flips 3 normal features and the "
        "curvature feature derived from them, on half of all training data."
    )
    assert np.isclose(left, right, atol=0.05), \
        f"left ({left:.4f}) and right ({right:.4f}) disagree on normal orientation"


def test_curvature_sign_survives_mirroring():
    """Our curvature feature is (mean(neighbours) - v) . n, a projection ONTO the
    normal, so it inherits any normal sign error. Convex must stay convex."""
    def concavity(mesh):
        V = np.asarray(mesh.vertices, dtype=float)
        F = np.asarray(mesh.faces)
        N = np.asarray(mesh.vertex_normals, dtype=float)
        acc = np.zeros_like(V)
        cnt = np.zeros(len(V))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = F[:, a], F[:, b]
            np.add.at(acc, i, V[j]); np.add.at(cnt, i, 1)
            np.add.at(acc, j, V[i]); np.add.at(cnt, j, 1)
        cnt = np.maximum(cnt, 1)[:, None]
        return float(np.einsum("ij,ij->i", acc / cnt - V, N).mean())

    with tempfile.TemporaryDirectory() as td:
        ds = _fake_dataset(Path(td))
        cl = concavity(load_canonical_mesh(ds, "P0001", "left"))
        cr = concavity(load_canonical_mesh(ds, "P0001", "right"))
    assert np.sign(cl) == np.sign(cr), (
        f"curvature sign flips between sides (left {cl:.5f}, right {cr:.5f}): "
        "convex surface would read as concave on right ears"
    )


# --------------------------------------------------------------- splitting

def test_splits_never_share_a_subject():
    """The leakage property. Left and right ears of one subject are the same
    anatomy mirrored, so a subject in both train and test would leak."""
    ids = [f"P{i:04d}" for i in range(1, 61)]
    for k in (2, 3, 4, 5):
        for train, val in k_fold_subject_split(ids, k, seed=0):
            assert not (set(train) & set(val)), f"subject leaked across a k={k} split"


def test_every_subject_is_tested_exactly_once():
    """Otherwise the reported mean is over an unknown, uneven sample."""
    ids = [f"P{i:04d}" for i in range(1, 61)]
    for k in (2, 3, 4, 5):
        seen = [s for _, val in k_fold_subject_split(ids, k, seed=0) for s in val]
        assert sorted(seen) == sorted(ids), f"k={k} does not cover every subject exactly once"


def test_splits_are_deterministic_for_a_seed():
    """A/B comparisons are only paired if both arms see identical splits."""
    ids = [f"P{i:04d}" for i in range(1, 41)]
    a = [(list(t), list(v)) for t, v in k_fold_subject_split(ids, 4, seed=7)]
    b = [(list(t), list(v)) for t, v in k_fold_subject_split(ids, 4, seed=7)]
    assert a == b, "same seed produced different splits"
    c = [(list(t), list(v)) for t, v in k_fold_subject_split(ids, 4, seed=8)]
    assert a != c, "different seeds produced identical splits -- seed is being ignored"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n        {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} Stage-1 provenance tests passed")
    sys.exit(1 if failed else 0)
