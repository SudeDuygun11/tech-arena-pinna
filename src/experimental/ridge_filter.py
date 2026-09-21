"""Connected ridge-corridor mesh filtering.

Motivation, measured (scripts/surface_recall.py, 24 ears): a plain curvature
percentile threshold is a POOR filter -- crest80 keeps 19.4% of faces but only
38.8% of ground-truth landmarks land within 0.25mm of the retained surface,
with p95 3.1mm and worst 5.7mm. The reason is visible when rendered: a bare
threshold yields scattered disconnected specks, not the continuous ridge band
the anatomy actually has.

Two steps fix that, and they are the whole idea:
  1. CONNECTIVITY -- keep only sufficiently large connected components of the
     thresholded set (discard the specks), computed over real mesh edges.
  2. DILATION -- grow the surviving components outward by a few edge rings, so
     the corridor is generous enough to contain the landmark rather than
     merely passing near it.

Deliberately NOT prediction-guided. A filter seeded on coarse predicted
landmarks inherits that model's mistakes as PERMANENT losses: a teammate's
prediction-seeded variants show worst-case 6.08mm, i.e. on some ear the true
landmark was deleted outright, and nothing downstream can recover it. Every
criterion here is a deterministic function of mesh geometry alone.


ROLE IN THE PIPELINE
--------------------
Reading order : 20 of 20   (experimental)
Duty          : Connected ridge corridor extraction.

Threshold, connected components, edge dilation. Part of the crest work that
measured a definitive null.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..correction.patch_features import _local_curvature

DEFAULT_PERCENTILE = 75.0
DEFAULT_MIN_COMPONENT = 60
DEFAULT_DILATE_RINGS = 3


def _vertex_adjacency(mesh) -> "coo_matrix":
    edges = mesh.edges_unique
    e0, e1 = edges[:, 0], edges[:, 1]
    n = len(mesh.vertices)
    data = np.ones(len(e0) * 2)
    rows = np.concatenate([e0, e1])
    cols = np.concatenate([e1, e0])
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def connected_ridge_vertices(mesh, percentile: float = DEFAULT_PERCENTILE,
                              min_component: int = DEFAULT_MIN_COMPONENT,
                              dilate_rings: int = DEFAULT_DILATE_RINGS,
                              curv: np.ndarray | None = None) -> np.ndarray:
    """Boolean vertex mask for the connected, dilated ridge corridor."""
    if curv is None:
        curv = _local_curvature(mesh, np.arange(len(mesh.vertices)))
    absc = np.abs(curv)
    seed = absc >= np.percentile(absc, percentile)
    if not seed.any():
        return seed

    adj = _vertex_adjacency(mesh)

    # --- step 1: connectivity. Restrict the graph to seed vertices so that
    # components are ridge fragments, then drop the small ones.
    e = mesh.edges_unique
    both = seed[e[:, 0]] & seed[e[:, 1]]
    n = len(mesh.vertices)
    sub = coo_matrix((np.ones(both.sum() * 2),
                      (np.concatenate([e[both, 0], e[both, 1]]),
                       np.concatenate([e[both, 1], e[both, 0]]))), shape=(n, n)).tocsr()
    n_comp, labels = connected_components(sub, directed=False)
    counts = np.bincount(labels[seed], minlength=n_comp)
    keep_labels = np.zeros(n_comp, dtype=bool)
    keep_labels[counts >= min_component] = True
    mask = seed & keep_labels[labels]
    if not mask.any():
        return mask

    # --- step 2: dilation along real mesh edges (geodesic, not euclidean)
    for _ in range(int(dilate_rings)):
        mask = mask | (adj.dot(mask.astype(np.float64)) > 0)
    return mask


def ridge_face_mask(mesh, min_votes: int = 2, **kwargs) -> np.ndarray:
    """Boolean FACE mask: a face is retained when >= min_votes of its corners
    are in the connected ridge corridor."""
    vmask = connected_ridge_vertices(mesh, **kwargs)
    return vmask[np.asarray(mesh.faces)].sum(axis=1) >= min_votes
