"""Is 'random beats medoid' for reference selection real, or one lucky seed?

The Tier 1 sweep tested exactly one random draw (seed 0) and one farthest-point
run (also deterministic, seeded from the medoid) against medoid-7. Random won
by -0.1026mm. Before trusting that over the "warp from a typical ear" reasoning
behind medoid selection, check whether it holds across several random seeds --
if the seed-to-seed spread is comparable to the medoid-vs-random gap, the
result is noise, not a finding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid, kabsch
from src.pipeline import predict_canonical
from src.registration import registration as R
from src.registration import template as T


def score(template, cache):
    errs = []
    for _, _, L, mesh in cache:
        pred = predict_canonical(template, None, mesh)
        errs.append(np.linalg.norm(pred - L, axis=1).mean())
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    train_ids = ds.subject_ids[:args.n_train]
    test_ids = ds.subject_ids[args.n_train:args.n_train + args.n_test]

    cache = [(e.subject_id, e.side, np.asarray(e.landmarks, float),
              load_canonical_mesh(ds, e.subject_id, e.side))
             for e in iter_landmarks_only(ds, test_ids)]
    print(f"{len(cache)} test ears cached\n", flush=True)

    ex = list(iter_landmarks_only(ds, train_ids))
    A = [np.asarray(e.landmarks, float)[ANCHOR_INDICES] for e in ex]
    mean = T._gpa_mean_anchor_shape(A)
    aligned = [apply_rigid(a, *kabsch(a, mean)) for a in A]
    res = np.array([np.mean(np.linalg.norm(al - mean, axis=1)) for al in aligned])

    def build_from_indices(idx):
        refs = []
        for i in idx:
            e = ex[i]
            V = np.asarray(load_canonical_mesh(ds, e.subject_id, e.side).vertices)
            c = e.landmarks.mean(axis=0)
            r = np.max(np.linalg.norm(e.landmarks - c, axis=1)) + T.CROP_MARGIN
            refs.append(R.build_reference(f"{e.subject_id}_{e.side}", e.landmarks, V, c, r))
        return R.Template(references=refs,
                          global_crop_center=base_tpl.global_crop_center,
                          global_crop_radius=base_tpl.global_crop_radius)

    base_tpl = T.build_template(ds, train_ids, k_references=args.k)
    base = score(base_tpl, cache)
    print(f"medoid-{args.k} (current design)      : {base:.4f} mm\n", flush=True)

    scores = []
    for seed in range(args.n_seeds):
        idx = np.random.default_rng(seed).choice(len(A), args.k, replace=False)
        tpl = build_from_indices(list(idx))
        s = score(tpl, cache)
        scores.append(s)
        print(f"  random seed {seed}  -> {s:.4f} mm  ({s-base:+.4f})", flush=True)

    scores = np.array(scores)
    print(f"\nrandom-{args.k}: mean {scores.mean():.4f}  sd {scores.std():.4f}  "
          f"min {scores.min():.4f}  max {scores.max():.4f}")
    print(f"medoid-{args.k}: {base:.4f}")
    print(f"\nmedoid beats random on {int((scores > base).sum())}/{len(scores)} seeds")
    print("If medoid falls outside [min,max] of random AND most seeds beat it,")
    print("the earlier 'random wins' result was likely one lucky seed.")


if __name__ == "__main__":
    main()
