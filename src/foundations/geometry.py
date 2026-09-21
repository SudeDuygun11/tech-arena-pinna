"""Rigid alignment, ear mirroring, and mesh-region cropping helpers.

The provided meshes are aligned along a common interaural axis (README):
Y runs left-ear -> right-ear, X back-to-front, Z bottom-to-top. This lets us
mirror a right ear into the left-ear canonical frame (negate Y) and train a
single shared left/right model.

ROLE IN THE PIPELINE
--------------------
Reading order : 2 of 20   (foundations)
Duty          : Geometric primitives shared by every stage.

kabsch/apply_rigid for Procrustes alignment; mirror_y for left/right
canonicalisation; nearest_surface_points for EXACT point-to-triangle
projection (k-nearest-centroid prefilter, then trimesh closest_point).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def mirror_y(points: np.ndarray) -> np.ndarray:
    """Reflect points across the sagittal (X-Z) plane, i.e. flip left<->right."""
    out = points.copy()
    out[..., 1] *= -1
    return out


def kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None):
    """Least-squares rigid transform (rotation + translation, no scale/reflection)
    mapping ``source`` points onto corresponding ``target`` points.

    Returns (R, t) such that target ~= source @ R.T + t
    """
    if weights is None:
        weights = np.ones(len(source))
    weights = weights / weights.sum()

    src_mean = (source * weights[:, None]).sum(axis=0)
    tgt_mean = (target * weights[:, None]).sum(axis=0)

    src_c = source - src_mean
    tgt_c = target - tgt_mean

    H = (src_c * weights[:, None]).T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    t = tgt_mean - src_mean @ R.T
    return R, t


def apply_rigid(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ R.T + t


def crop_region(vertices: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Boolean mask of vertices within `radius` of `center`."""
    dist = np.linalg.norm(vertices - center[None, :], axis=1)
    return dist <= radius


def nearest_vertices(vertices: np.ndarray, query_points: np.ndarray, k: int = 1):
    tree = cKDTree(vertices)
    dist, idx = tree.query(query_points, k=k)
    return dist, idx


def nearest_surface_points(vertices: np.ndarray, faces: np.ndarray, query: np.ndarray,
                            k: int = 24):
    """Exact closest point ON THE TRIANGLE SURFACE for each query point.

    Distinct from nearest_vertices: a landmark can sit in the middle of a large
    triangle, far from any of its corners, so vertex distance systematically
    OVERSTATES how far a point is from the surface. That difference is why our
    own vertex-based crest measurement (~0.8-1.1mm) is not comparable to a
    triangle-based one (~0.04mm p95) -- they measure different things.

    Exact rather than approximate: k nearest triangles are picked by centroid
    (cheap prefilter, no rtree dependency), then the true point-to-triangle
    closest point is computed for each candidate and the best kept.

    Returns (closest_xyz (N,3), distance (N,)).
    """
    import trimesh
    from scipy.spatial import cKDTree as _KD

    if len(faces) == 0:
        return np.full_like(query, np.nan), np.full(len(query), np.inf)

    tris = vertices[faces]                    # (F, 3, 3)
    centroids = tris.mean(axis=1)             # (F, 3)
    k = int(min(k, len(faces)))
    _, cand = _KD(centroids).query(query, k=k)
    cand = np.atleast_2d(cand.T).T if k > 1 else cand.reshape(-1, 1)

    n = len(query)
    # evaluate all (point, candidate-triangle) pairs in one vectorised call
    pts_rep = np.repeat(query, k, axis=0)                 # (n*k, 3)
    tri_rep = tris[cand.reshape(-1)]                      # (n*k, 3, 3)
    closest = trimesh.triangles.closest_point(tri_rep, pts_rep)
    d = np.linalg.norm(closest - pts_rep, axis=1).reshape(n, k)

    best = np.argmin(d, axis=1)
    rows = np.arange(n)
    return closest.reshape(n, k, 3)[rows, best], d[rows, best]
