"""Is landmark 75 directly findable from curvature?

Hypothesis (user observation, backed by the organizer's own definition):
index 75 is the "junction of crura of antihelix and outer concha contour" --
anatomically the point where the antihelix bifurcates into its two crura.
That is a visually obvious peaked ridge feature ("tepeli"), so it should sit
at a curvature EXTREMUM rather than needing to be learned from a patch.

If true, 75 can be snapped to a curvature feature instead of regressed.
It is worth -0.0589mm on its own, and it is the THIRD anchor in the greedy
route to 1.1mm (74 -> 1.2091, +64 -> 1.1318, +75 -> 1.0718).

What is measured, per ear:
  1. The curvature percentile rank of the mesh vertex nearest GT 75. If 75 is
     a curvature extremum this should sit near 100 (ridge) or near 0 (valley).
     A rank near 50 would falsify the hypothesis outright.
  2. ORACLE SNAP: within radius r of GT 75, how close is the most extreme
     curvature vertex? This is the best any curvature-snapping rule could do.
  3. REAL SNAP: within radius r of OUR PREDICTED 75, snap to the most extreme
     curvature vertex and measure the resulting error against GT. This is what
     the rule would actually deliver, and the only number that decides it.

Both mean (Laplacian . normal) and discrete Gaussian curvature are tried --
a bifurcation may well be a SADDLE (negative Gaussian) rather than a simple
ridge, and those are different detectors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correction.patch_features import _gaussian_curvature, _local_curvature, crop_submesh
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.dataset import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--preds", default="results/twopass_surfsnap.npz")
    ap.add_argument("--n-ears", type=int, default=40)
    ap.add_argument("--radii", type=float, nargs="+", default=[1.5, 2.5, 4.0, 6.0])
    ap.add_argument("--landmark", type=int, default=75)
    ap.add_argument("--crop-radius", type=float, default=14.0)
    args = ap.parse_args()

    gi = args.landmark
    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))

    d = np.load(args.preds, allow_pickle=True)
    pred_all, truth_all = d["pred"], d["truth"]
    sids = np.array([str(s) for s in d["subject_id"]])
    sides = np.array([str(s) for s in d["side"]])
    lookup = {(sids[i], sides[i]): i for i in range(len(sids))}

    print(f"landmark {gi}; {args.n_ears} ears; crop radius {args.crop_radius}mm", flush=True)

    ranks = {"mean": [], "gauss": []}
    oracle = {("mean", r): [] for r in args.radii}
    oracle.update({("gauss", r): [] for r in args.radii})
    real = {k: [] for k in oracle}
    baseline = []

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
        V = np.asarray(sub.vertices)
        if len(V) < 200:
            continue
        idx = np.arange(len(V))
        curv = {"mean": _local_curvature(sub, idx), "gauss": _gaussian_curvature(sub, idx)}
        tree = cKDTree(V)

        baseline.append(np.linalg.norm(pred[gi] - truth[gi]))
        for kind, c in curv.items():
            # 1. percentile rank of |curvature| at the true landmark
            j = tree.query(truth[gi])[1]
            ranks[kind].append(100.0 * (np.abs(c) < abs(c[j])).mean())
            for r in args.radii:
                # 2. oracle: most extreme |curvature| vertex near TRUE 75
                nb = tree.query_ball_point(truth[gi], r)
                if nb:
                    k = nb[int(np.argmax(np.abs(c[nb])))]
                    oracle[(kind, r)].append(np.linalg.norm(V[k] - truth[gi]))
                # 3. real: most extreme |curvature| vertex near OUR 75
                nb = tree.query_ball_point(pred[gi], r)
                if nb:
                    k = nb[int(np.argmax(np.abs(c[nb])))]
                    real[(kind, r)].append(np.linalg.norm(V[k] - truth[gi]))
        done += 1
        if done % 10 == 0:
            print(f"  {done} ears...", flush=True)
        if done >= args.n_ears:
            break

    b = float(np.mean(baseline))
    print(f"\nour current error on landmark {gi}: {b:.4f} mm   (n={done} ears)\n")

    print("1. curvature percentile rank AT the true landmark (50 = unremarkable):")
    for kind in ("mean", "gauss"):
        v = np.array(ranks[kind])
        print(f"   {kind:6s}  {v.mean():5.1f}  (sd {v.std():4.1f}, median {np.median(v):5.1f})")

    print("\n2/3. snap to the most extreme |curvature| vertex within radius r:")
    print(f"   {'':6s} {'r':>5s} {'ORACLE (near truth)':>21s} {'REAL (near our pred)':>22s}")
    for kind in ("mean", "gauss"):
        for r in args.radii:
            o = np.mean(oracle[(kind, r)]) if oracle[(kind, r)] else float("nan")
            a = np.mean(real[(kind, r)]) if real[(kind, r)] else float("nan")
            flag = "  <-- beats current" if a < b else ""
            print(f"   {kind:6s} {r:5.1f} {o:21.4f} {a:22.4f}{flag}")

    print(f"\nREAD: column 1 near 50 falsifies the hypothesis. A REAL number below")
    print(f"{b:.4f} means curvature snapping beats our learned prediction for this point.")


if __name__ == "__main__":
    main()
