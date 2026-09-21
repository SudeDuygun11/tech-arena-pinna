"""Landmark 75 at the scale a human actually sees it.

The one-ring test (lm75_curvature_test.py) found no curvature signal at 75:
percentile rank 56.5 (mean curv) / 25.2 (Gaussian), and snapping was worse
than our prediction at every radius. That test used a one-ring Laplacian on a
~574k-vertex scan with sub-millimetre triangles, so it measures scan noise,
not anatomy.

The antihelix bifurcation -- "junction of crura of antihelix and outer concha
contour" -- is a feature at 2-5mm. That is the scale the eye integrates over
when a person picks the spot by looking. This runs the same test at a SWEEP
of scales.

Scale-sigma mean curvature, the natural generalisation of the one-ring
Laplacian: for each vertex v with smoothed normal n,
    H_sigma(v) = ((centroid of vertices within sigma) - v) . n / sigma^2
The normal is itself re-estimated at scale sigma by PCA of the same ball, so
both the normal and the displacement are smoothed consistently.

Also tested: SADDLE-ness. A ridge bifurcation is where one crest line splits,
which is a saddle, not a peak -- there the two principal curvatures have
opposite signs and scale-Gaussian curvature goes NEGATIVE. A detector looking
for |curvature| maxima would miss that entirely, which may be why the
one-ring Gaussian rank came out at 25 rather than 75.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correction.patch_features import crop_submesh
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.dataset import Dataset


def scale_curvature(V: np.ndarray, sigma: float):
    """Return (H, K_proxy) at scale sigma: mean-curvature proxy and a
    saddle indicator, both computed from a PCA of each sigma-ball."""
    tree = cKDTree(V)
    nb = tree.query_ball_point(V, sigma)
    H = np.zeros(len(V))
    K = np.zeros(len(V))
    for i, ids in enumerate(nb):
        if len(ids) < 6:
            continue
        P = V[ids]
        c = P.mean(0)
        C = np.cov((P - c).T)
        w, U = np.linalg.eigh(C)          # ascending; U[:,0] ~ normal
        nrm = U[:, 0]
        H[i] = np.dot(c - V[i], nrm) / (sigma * sigma)
        # curvature anisotropy in the tangent plane: a saddle has the two
        # tangent extents comparable but the surface bending both ways, which
        # shows up as a large normal-direction variance relative to the ball.
        K[i] = w[0] / max(w[1], 1e-12)
    return H, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--preds", default="results/twopass_surfsnap.npz")
    ap.add_argument("--n-ears", type=int, default=25)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--radii", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--landmark", type=int, default=75)
    ap.add_argument("--crop-radius", type=float, default=12.0)
    ap.add_argument("--decimate", type=int, default=4,
                    help="keep 1/N vertices; the whole point is coarse scale")
    args = ap.parse_args()

    gi = args.landmark
    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    d = np.load(args.preds, allow_pickle=True)
    pred_all, truth_all = d["pred"], d["truth"]
    sids = np.array([str(s) for s in d["subject_id"]])
    sides = np.array([str(s) for s in d["side"]])
    lookup = {(sids[i], sides[i]): i for i in range(len(sids))}

    print(f"landmark {gi}; {args.n_ears} ears; sigmas {args.sigmas}", flush=True)
    rank_H = {s: [] for s in args.sigmas}
    rank_signed = {s: [] for s in args.sigmas}
    real = {(s, r): [] for s in args.sigmas for r in args.radii}
    orac = {(s, r): [] for s in args.sigmas for r in args.radii}
    base = []
    done = 0

    for e in iter_landmarks_only(ds, ds.subject_ids):
        key = (e.subject_id, e.side)
        if key not in lookup:
            continue
        i = lookup[key]
        truth, pred = truth_all[i], pred_all[i]
        mesh = load_canonical_mesh(ds, e.subject_id, e.side)
        sub = crop_submesh(mesh, truth[gi], args.crop_radius)
        del mesh
        V = np.asarray(sub.vertices)[::args.decimate]
        if len(V) < 300:
            continue
        base.append(np.linalg.norm(pred[gi] - truth[gi]))
        tree = cKDTree(V)
        jt = tree.query(truth[gi])[1]
        for s in args.sigmas:
            H, _ = scale_curvature(V, s)
            rank_H[s].append(100.0 * (np.abs(H) < abs(H[jt])).mean())
            rank_signed[s].append(100.0 * (H < H[jt]).mean())
            for r in args.radii:
                nb = tree.query_ball_point(truth[gi], r)
                if nb:
                    k = nb[int(np.argmax(np.abs(H[nb])))]
                    orac[(s, r)].append(np.linalg.norm(V[k] - truth[gi]))
                nb = tree.query_ball_point(pred[gi], r)
                if nb:
                    k = nb[int(np.argmax(np.abs(H[nb])))]
                    real[(s, r)].append(np.linalg.norm(V[k] - truth[gi]))
        done += 1
        if done % 5 == 0:
            print(f"  {done} ears...", flush=True)
        if done >= args.n_ears:
            break

    b = float(np.mean(base))
    print(f"\nour current error on landmark {gi}: {b:.4f} mm  (n={done})\n")
    print("percentile rank of the TRUE landmark, by scale (50 = unremarkable):")
    print(f"  {'sigma':>6s} {'|H| rank':>10s} {'signed H rank':>14s}")
    for s in args.sigmas:
        print(f"  {s:6.1f} {np.mean(rank_H[s]):10.1f} {np.mean(rank_signed[s]):14.1f}")
    print("\n  signed rank near 0 or 100 = a consistent VALLEY or RIDGE at that scale,")
    print("  even if |H| rank is unremarkable.")

    print("\nsnap to max |H| within radius r:")
    print(f"  {'sigma':>6s} {'r':>5s} {'ORACLE':>9s} {'REAL':>9s}")
    for s in args.sigmas:
        for r in args.radii:
            o = np.mean(orac[(s, r)]); a = np.mean(real[(s, r)])
            print(f"  {s:6.1f} {r:5.1f} {o:9.4f} {a:9.4f}"
                  f"{'   <-- beats current' if a < b else ''}")


if __name__ == "__main__":
    main()
