"""Apply the organizer's rule for landmark 6 to the MESH ALONE.

The chain 6 -> 64 -> 74 is exact on true curves (median 0.047mm / 0.417 /
0.120). It fails on our own curves because those are built from the very
landmarks the rule is meant to find -- pure circularity.

The escape: the helix rim is a physical ridge on the scan. A curve derived
from surface geometry owes nothing to our landmark predictions, so the rule
can be evaluated honestly. This tests exactly that, using NO landmark
information as input.

"Index 6: upper point of the largest extent of the outer helix" is
operationalised as the upper endpoint of the maximum-diameter chord of a
geometric point set. Three candidate sets, in increasing sophistication:

  A. every vertex of the cropped ear      -- the crudest possible reading:
     the outer helix IS the ear's outline, so the ear's own widest chord may
     already be 6 <-> 22.
  B. ridge-corridor vertices (connected_ridge_vertices, geometry-only)
  C. ridge corridor at a coarser curvature scale, since the one-ring
     Laplacian was shown to measure scan noise rather than anatomy.

Scored against true 6 and true 22. Anything approaching the 1.599 / 1.956 mm
our learned model achieves would mean the rule can be evaluated from geometry
alone -- which is the whole question.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correction.patch_features import crop_submesh
from src.experimental.ridge_filter import connected_ridge_vertices
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.dataset import Dataset


def max_diameter_pair(P, cap=4000, seed=0):
    """Upper/lower endpoints of the widest chord of a point set."""
    if len(P) > cap:
        P = P[np.random.default_rng(seed).choice(len(P), cap, replace=False)]
    # two-pass farthest-point: exact enough for a convex-ish outline and O(n)
    a = P[np.argmax(np.linalg.norm(P - P.mean(0), axis=1))]
    b = P[np.argmax(np.linalg.norm(P - a, axis=1))]
    c = P[np.argmax(np.linalg.norm(P - b, axis=1))]
    p, q = (b, c) if np.linalg.norm(b - c) > np.linalg.norm(a - b) else (a, b)
    return (p, q) if p[2] > q[2] else (q, p)


def scale_curv(V, sigma=2.5):
    tree = cKDTree(V)
    out = np.zeros(len(V))
    for i, ids in enumerate(tree.query_ball_point(V, sigma)):
        if len(ids) < 6:
            continue
        Q = V[ids]; c = Q.mean(0)
        w, U = np.linalg.eigh(np.cov((Q - c).T))
        out[i] = np.dot(c - V[i], U[:, 0]) / (sigma * sigma)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-ears", type=int, default=20)
    ap.add_argument("--crop-radius", type=float, default=45.0)
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    res = {k: {6: [], 22: []} for k in ("A_all_vertices", "B_ridge", "C_ridge_coarse")}
    done = 0

    for e in iter_landmarks_only(ds, ds.subject_ids):
        L = np.asarray(e.landmarks, float)
        mesh = load_canonical_mesh(ds, e.subject_id, e.side)
        centre = L[0:25].mean(0)
        sub = crop_submesh(mesh, centre, args.crop_radius)
        del mesh
        V = np.asarray(sub.vertices)
        if len(V) < 1000:
            continue

        sets = {"A_all_vertices": V}
        try:
            m = connected_ridge_vertices(sub)
            if m.sum() > 200:
                sets["B_ridge"] = V[m]
        except Exception:
            pass
        Vd = V[::3]
        c = scale_curv(Vd, 2.5)
        keep = np.abs(c) >= np.percentile(np.abs(c), 75)
        if keep.sum() > 200:
            sets["C_ridge_coarse"] = Vd[keep]

        for k, P in sets.items():
            up, lo = max_diameter_pair(P)
            res[k][6].append(np.linalg.norm(up - L[6]))
            res[k][22].append(np.linalg.norm(lo - L[22]))

        done += 1
        if done % 5 == 0:
            print(f"  {done} ears...", flush=True)
        if done >= args.n_ears:
            break

    print(f"\nMax-diameter rule applied to MESH GEOMETRY ONLY (no landmarks in).")
    print(f"Our learned predictions, for comparison: 6 = 1.5992, 22 = 1.9558 mm\n")
    print(f"  {'point set':20s} {'lm 6 mean':>10s} {'median':>8s} {'lm 22 mean':>11s} {'median':>8s}")
    for k in ("A_all_vertices", "B_ridge", "C_ridge_coarse"):
        a, b = np.array(res[k][6]), np.array(res[k][22])
        if not len(a):
            print(f"  {k:20s} {'--':>10s}")
            continue
        print(f"  {k:20s} {a.mean():10.4f} {np.median(a):8.4f} "
              f"{b.mean():11.4f} {np.median(b):8.4f}")


if __name__ == "__main__":
    main()
