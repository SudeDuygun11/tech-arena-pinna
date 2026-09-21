"""Local surface-patch feature extraction around a candidate anchor location.

Used both to train and to run the Stage-1 residual-correction network: crop
a small neighborhood around the registration-predicted anchor position and
describe it with per-point relative position, normal, and a fast curvature
proxy (uniform-Laplacian projected onto the normal -- positive on ridges,
negative in grooves/sulci).


ROLE IN THE PIPELINE
--------------------
Reading order : 8 of 20   (correction)
Duty          : Stage 1b input: a 3D neighbourhood as a fixed tensor.

All vertices within a 7.0mm Euclidean ball, 7 features each: rel_xyz/radius
(3), vertex normal (3), local curvature (1). Subsampled to 256 points at
random rather than by farthest-point, because randomness doubles as
augmentation and enables test-time averaging over resamples.

Known issues / status:
  Zero-padding is semantically ambiguous: a pad row reads as a real point AT
  the patch centre with a zero normal. About 3 percent of patches normally, 17
  percent with geodesic balls, which is one reason that experiment was
  confounded.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

PATCH_RADIUS = 7.0
PATCH_N_POINTS = 256

# Multi-scale variant: split the same 256-point budget across a fine (ridge
# detail) and coarse (broader concha/contour context) radius instead of one
# fixed radius, to test whether more informative input helps an underfit
# model -- kept at the same total point count as the single-scale default so
# any improvement isolates "multi-scale info" from "more points".
PATCH_RADIUS_FINE = 7.0
PATCH_RADIUS_COARSE = 14.0
PATCH_N_POINTS_FINE = 128
PATCH_N_POINTS_COARSE = 128


def crop_submesh(mesh: trimesh.Trimesh, center: np.ndarray, radius: float) -> trimesh.Trimesh:
    """A much smaller mesh containing only the region around `center`. Extracting
    many small patches from this instead of the full (300k+ vertex) head mesh
    avoids repeating an O(n_vertices) distance scan for every single patch."""
    dist = np.linalg.norm(mesh.vertices - center[None, :], axis=1)
    keep_mask = dist <= radius
    face_mask = keep_mask[mesh.faces].any(axis=1)
    if not face_mask.any():
        return mesh
    return mesh.submesh([face_mask], append=True)


def _gaussian_curvature(mesh: trimesh.Trimesh, keep_idx: np.ndarray) -> np.ndarray:
    """Discrete Gaussian curvature via the angle-deficit (Gauss-Bonnet) formula:
    K_i = 2*pi - sum of incident face corner angles at vertex i.

    Complements _local_curvature (a mean-curvature-like ridge/valley signal):
    mean curvature cannot distinguish a saddle from a flat region (both can
    average to ~0), whereas Gaussian curvature is negative on saddles, positive
    on domes/bowls, ~0 on developable surfaces -- directly useful for telling
    the concha bowl (positive) from the helix rim's saddle-like transitions.

    Uses trimesh's own vectorised corner angles, so this stays fast (unlike
    trimesh.curvature.discrete_gaussian_curvature_measure, which needs rtree
    and was measured at ~134s for 28k points earlier in this project).
    """
    faces = mesh.faces
    angles = mesh.face_angles  # (n_faces, 3), corner angle at each face vertex
    defect = np.full(len(mesh.vertices), 2.0 * np.pi)
    np.subtract.at(defect, faces.ravel(), angles.ravel())
    return defect[keep_idx]


def _local_curvature(mesh: trimesh.Trimesh, keep_idx: np.ndarray) -> np.ndarray:
    keep_mask = np.zeros(len(mesh.vertices), dtype=bool)
    keep_mask[keep_idx] = True
    remap = -np.ones(len(mesh.vertices), dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))

    edges = mesh.edges_unique
    e0, e1 = edges[:, 0], edges[:, 1]
    both = keep_mask[e0] & keep_mask[e1]
    r0, r1 = remap[e0[both]], remap[e1[both]]

    n = len(keep_idx)
    adj = coo_matrix((np.ones(len(r0) * 2), (np.concatenate([r0, r1]), np.concatenate([r1, r0]))),
                      shape=(n, n)).tocsr()
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1

    sub_v = mesh.vertices[keep_idx]
    sub_n = mesh.vertex_normals[keep_idx]
    neighbor_avg = adj.dot(sub_v) / deg[:, None]
    laplacian = neighbor_avg - sub_v
    return np.einsum("ij,ij->i", laplacian, sub_n)


def geodesic_ball(mesh, center: np.ndarray, radius: float) -> np.ndarray:
    """Vertex indices within `radius` GEODESIC (along-surface) distance of the
    vertex nearest `center`.

    Why this matters for ears specifically: the helix FOLDS OVER ITSELF. A point
    on the rim and a point on the inside of that fold can be ~3mm apart in
    straight-line distance but ~15mm apart along the surface. A Euclidean ball
    therefore mixes geometry from the wrong side of a fold into the patch, which
    is noise the network has to learn to ignore -- and cannot distinguish from
    the ridge it is supposed to be reading.

    Falls back to the Euclidean ball if the mesh has no usable edges.
    """
    V = np.asarray(mesh.vertices)
    edges = mesh.edges_unique
    if len(edges) == 0:
        return np.where(np.linalg.norm(V - center[None, :], axis=1) <= radius)[0]

    src = int(cKDTree(V).query(center[None, :])[1][0])
    w = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    n = len(V)
    g = coo_matrix((np.concatenate([w, w]),
                    (np.concatenate([edges[:, 0], edges[:, 1]]),
                     np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(n, n)).tocsr()
    d = dijkstra(g, indices=src, limit=radius)
    return np.where(np.isfinite(d) & (d <= radius))[0]


# Input channels zeroed for ablation. Zeroing (rather than deleting) keeps every
# tensor shape, the model constructor and the saved-bundle format unchanged, so a
# feature-ablation run differs from the baseline in exactly one respect: the
# network receives no information through the dropped channels.
#
# Channel layout of extract_patch output:
#   0-2 rel_xyz/radius   3-5 normal   6 mean curvature   7 Gaussian (if enabled)
# MUST be set identically for training and inference -- cross_validate.py records
# it in the bundle config and estimator.py re-applies it on load.
_FEATURE_SLICES = {"xyz": slice(0, 3), "normal": slice(3, 6), "curv": slice(6, 7),
                   "gauss": slice(7, 8)}
_DROPPED: tuple = ()

# Bumped whenever patch contents change, and part of the example-cache key.
FEATURE_VERSION = "2026-09-15-curvclip-v2"
# One-ring curvature is typically 0.0115 (p99.9 0.092) but reaches 70 on
# degenerate mesh geometry; unclipped, one such spike made a trained network
# output a 150mm correction. Clip well above every legitimate value.
CURV_CLIP = 0.25
GAUSS_CLIP = 0.5

_POINT_FEATURE_MODES = ("base", "curvedness", "rich")
_POINT_FEATURES = "base"


def set_point_features(mode: str) -> None:
    global _POINT_FEATURES
    if mode not in _POINT_FEATURE_MODES:
        raise ValueError(f"point feature mode {mode!r} not in {_POINT_FEATURE_MODES}")
    _POINT_FEATURES = mode


def point_features() -> str:
    return _POINT_FEATURES


def _mesh_tree(mesh):
    tree = mesh.metadata.get("_pf_tree")
    if tree is None:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(mesh.vertices))
        mesh.metadata["_pf_tree"] = tree
    return tree


def _surface_descriptors(mesh, idx: np.ndarray, rich: bool) -> np.ndarray:
    """Shape index and curvedness at ~2mm from a batched quadric fit over 48
    neighbours (Koenderink & van Doorn 1992; used for 3D ear helix/antihelix
    detection by Chen & Bhanu 2007); with rich=True also the ridge-direction
    tensor (principal direction of the larger curvature, sign-free) and fold
    enclosure (share of surface within 6mm lying >1mm above the tangent plane)."""
    V = np.asarray(mesh.vertices); N = np.asarray(mesh.vertex_normals)
    tree = _mesh_tree(mesh)
    k = int(min(48, len(V)))
    _, nb = tree.query(V[idx], k=k)
    n = N[idx]
    ref = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    t1 = ref - (ref * n).sum(1, keepdims=True) * n
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(n, t1)
    P = V[nb] - V[idx][:, None, :]
    x = (P * t1[:, None, :]).sum(2); y = (P * t2[:, None, :]).sum(2); z = (P * n[:, None, :]).sum(2)
    A = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=2)
    AtA = np.einsum("nki,nkj->nij", A, A) + 1e-9 * np.eye(6)
    coef = np.linalg.solve(AtA, np.einsum("nki,nk->ni", A, z)[..., None])[..., 0]
    H = np.stack([np.stack([2 * coef[:, 0], coef[:, 1]], 1), np.stack([coef[:, 1], 2 * coef[:, 2]], 1)], 1)
    w, U = np.linalg.eigh(H)
    k2, k1 = w[:, 0], w[:, 1]
    si = (2.0 / np.pi) * np.arctan2(k1 + k2, np.abs(k1 - k2) + 1e-12)
    cv = np.clip(3.0 * np.sqrt((k1 ** 2 + k2 ** 2) / 2.0), 0.0, 3.0)
    cols = [si, cv]
    if rich:
        big = (np.abs(k1) >= np.abs(k2)).astype(int)
        u = U[np.arange(len(idx)), :, big]
        d = u[:, :1] * t1 + u[:, 1:2] * t2
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        cols += [d[:, 0] ** 2, d[:, 1] ** 2, d[:, 2] ** 2, d[:, 0] * d[:, 1], d[:, 0] * d[:, 2], d[:, 1] * d[:, 2]]
        dd, nb6 = tree.query(V[idx], k=int(min(160, len(V))), distance_upper_bound=6.0)
        valid = np.isfinite(dd); nb6c = np.where(valid, nb6, 0)
        hgt = ((V[nb6c] - V[idx][:, None, :]) * n[:, None, :]).sum(2)
        cols.append(((hgt > 1.0) & valid).sum(1) / np.maximum(valid.sum(1), 1))
    return np.column_stack(cols).astype(np.float32)


def set_dropped_features(names) -> None:
    global _DROPPED
    names = tuple(n for n in (names or ()) if n)
    unknown = [n for n in names if n not in _FEATURE_SLICES]
    if unknown:
        raise ValueError(f"unknown feature channel(s) {unknown}; "
                         f"choose from {sorted(_FEATURE_SLICES)}")
    _DROPPED = names


def dropped_features() -> tuple:
    return _DROPPED


def extract_patch(mesh: trimesh.Trimesh, center: np.ndarray, radius: float = PATCH_RADIUS,
                   n_points: int = PATCH_N_POINTS, rng: np.random.Generator | None = None,
                   gaussian: bool = False, geodesic: bool = False) -> np.ndarray:
    """Returns (n_points, 7) features: rel_xyz/radius (3), normal (3), curvature (1),
    or (n_points, 8) with `gaussian=True`, appending discrete Gaussian curvature.
    Zero-padded if fewer than n_points vertices fall in the patch.

    The trailing channels depend on set_point_features(): 'base' (one-ring
    curvature, clipped), 'curvedness' (shape index + curvedness), 'rich'
    (+ ridge-direction tensor + fold enclosure). See _surface_descriptors."""
    mode = _POINT_FEATURES
    dim = 6 + {"base": 1, "curvedness": 2, "rich": 9}[mode] + (1 if gaussian else 0)
    if geodesic:
        keep_idx = geodesic_ball(mesh, center, radius)
    else:
        dist = np.linalg.norm(mesh.vertices - center[None, :], axis=1)
        keep_idx = np.where(dist <= radius)[0]
    if len(keep_idx) == 0:
        return np.zeros((n_points, dim), dtype=np.float32)

    if mode != "base":
        if any(d not in ("xyz", "normal") for d in _DROPPED):
            raise ValueError("--drop-features curv/gauss only apply to point-features 'base'")
        rng = rng or np.random.default_rng()
        if len(keep_idx) > n_points:
            keep_idx = rng.choice(keep_idx, n_points, replace=False)
        parts = [(mesh.vertices[keep_idx] - center[None, :]) / radius, mesh.vertex_normals[keep_idx],
                 _surface_descriptors(mesh, keep_idx, rich=(mode == "rich"))]
        if gaussian:
            parts.append(np.clip(_gaussian_curvature(mesh, keep_idx), -GAUSS_CLIP, GAUSS_CLIP)[:, None])
        feats = np.concatenate(parts, axis=1).astype(np.float32)
        for name in _DROPPED:
            feats[:, _FEATURE_SLICES[name]] = 0.0
        if len(feats) < n_points:
            feats = np.concatenate([feats, np.zeros((n_points - len(feats), dim), np.float32)], axis=0)
        return feats

    curvature = np.clip(_local_curvature(mesh, keep_idx), -CURV_CLIP, CURV_CLIP)
    rel_xyz = (mesh.vertices[keep_idx] - center[None, :]) / radius
    normals = mesh.vertex_normals[keep_idx]
    parts = [rel_xyz, normals, curvature[:, None]]
    if gaussian:
        parts.append(np.clip(_gaussian_curvature(mesh, keep_idx), -GAUSS_CLIP, GAUSS_CLIP)[:, None])
    feats = np.concatenate(parts, axis=1).astype(np.float32)
    for name in _DROPPED:
        sl = _FEATURE_SLICES[name]
        if sl.start < feats.shape[1]:
            feats[:, sl] = 0.0

    if len(feats) >= n_points:
        rng = rng or np.random.default_rng()
        sel = rng.choice(len(feats), n_points, replace=False)
        feats = feats[sel]
    else:
        pad = np.zeros((n_points - len(feats), dim), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=0)
    return feats


def extract_multiscale_patch(mesh: trimesh.Trimesh, center: np.ndarray,
                              radius_fine: float = PATCH_RADIUS_FINE,
                              radius_coarse: float = PATCH_RADIUS_COARSE,
                              n_points_fine: int = PATCH_N_POINTS_FINE,
                              n_points_coarse: int = PATCH_N_POINTS_COARSE,
                              rng: np.random.Generator | None = None) -> np.ndarray:
    """Returns (n_points_fine + n_points_coarse, 7) features: a fine-radius patch
    (rel_xyz normalized by radius_fine) concatenated with a coarse-radius patch
    (rel_xyz normalized by radius_coarse), giving the network both local ridge
    detail and broader contour context from the same point budget."""
    rng = rng or np.random.default_rng()
    fine = extract_patch(mesh, center, radius=radius_fine, n_points=n_points_fine, rng=rng)
    coarse = extract_patch(mesh, center, radius=radius_coarse, n_points=n_points_coarse, rng=rng)
    return np.concatenate([fine, coarse], axis=0)
