"""Pre-build measurement for the candidate-generation model (Option B).

Answers three questions BEFORE any ranker is written, because if the candidate
set does not contain the answer, no amount of ranking can recover it:

  1. How far is the registration SEED from truth? (sets the minimum radius)
  2. How many mesh vertices fall within a given radius? (sets the count)
  3. Is truth close to SOME candidate, on EVERY ear? (recall / worst case)

Leakage-free: the registration template is built from subjects disjoint from
the ones measured.

Reports, per radius:
  - mean/p95 vertex count in the ball
  - MISS RATE: fraction of anchors whose truth lies OUTSIDE the ball entirely
    (catastrophic -- unrecoverable by the ranker)
  - recall at several tolerances: truth within X mm of the nearest candidate
  - worst-case nearest-candidate distance
Both for all vertices in the ball, and after subsampling to a fixed budget.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES
from src.foundations.dataset import Dataset
from src.correction.patch_features import crop_submesh
from src.registration.registration import compute_raw_full
from src.foundations.splits import k_fold_subject_split
from src.registration.template import build_template

RADII = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
TOLERANCES = [0.25, 0.50, 1.00]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=24)
    p.add_argument("--k-references", type=int, default=7)
    p.add_argument("--budget", type=int, default=128,
                    help="fixed candidate count to subsample to")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    ids = ds.subject_ids[:args.n_subjects]

    # leakage-free: template from one half, measure on the other
    (train_ids, test_ids) = next(iter(k_fold_subject_split(ids, 2, args.seed)))
    print(f"template from {len(train_ids)} subjects; measuring on {len(test_ids)}")
    t0 = time.time()
    template = build_template(ds, train_ids, k_references=args.k_references)
    print(f"template built in {time.time() - t0:.1f}s\n")

    seed_err = []
    counts = {r: [] for r in RADII}
    miss = {r: [] for r in RADII}          # truth outside the ball entirely
    near_all = {r: [] for r in RADII}      # truth -> nearest candidate (all in ball)
    near_sub = {r: [] for r in RADII}      # ... after subsampling to budget
    rng = np.random.default_rng(args.seed)
    n_ears = 0

    for ex in iter_landmarks_only(ds, test_ids):
        mesh = load_canonical_mesh(ds, ex.subject_id, ex.side)
        raw = compute_raw_full(template, mesh.vertices, method="tps")
        local = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)
        V = np.asarray(local.vertices)
        n_ears += 1

        for a in ANCHOR_INDICES:
            s, t = raw[a], ex.landmarks[a]
            seed_err.append(float(np.linalg.norm(s - t)))
            d_from_seed = np.linalg.norm(V - s[None, :], axis=1)
            for r in RADII:
                inside = d_from_seed <= r
                counts[r].append(int(inside.sum()))
                if not inside.any():
                    miss[r].append(True)
                    near_all[r].append(np.inf)
                    near_sub[r].append(np.inf)
                    continue
                cand = V[inside]
                dt = np.linalg.norm(cand - t[None, :], axis=1)
                near_all[r].append(float(dt.min()))
                # did the true landmark fall outside the ball's reach at all?
                miss[r].append(bool(np.linalg.norm(s - t) > r))
                if len(cand) > args.budget:
                    sel = rng.choice(len(cand), args.budget, replace=False)
                    near_sub[r].append(float(dt[sel].min()))
                else:
                    near_sub[r].append(float(dt.min()))

    se = np.array(seed_err)
    print(f"ears: {n_ears}   anchors measured: {len(se)}\n")
    print("=== registration SEED error (raw TPS, before any correction) ===")
    print(f"  mean {se.mean():.3f}mm   median {np.median(se):.3f}mm   "
          f"p95 {np.percentile(se, 95):.3f}mm   max {se.max():.3f}mm\n")

    hdr = (f"{'radius':>7s}{'n_vtx mean':>12s}{'n_vtx p95':>11s}{'MISS%':>8s}"
           + "".join(f"rec<={t:.2f}".rjust(10) for t in TOLERANCES)
           + f"{'worst':>10s}")
    print("=== candidates = ALL vertices in ball ===")
    print(hdr)
    print("-" * len(hdr))
    for r in RADII:
        d = np.array(near_all[r]); c = np.array(counts[r]); m = np.array(miss[r])
        finite = d[np.isfinite(d)]
        row = (f"{r:6.1f}m{c.mean():11.0f}{np.percentile(c, 95):11.0f}"
               f"{m.mean() * 100:7.2f}%")
        for t in TOLERANCES:
            row += f"{(d <= t).mean() * 100:9.2f}%"
        row += f"{(finite.max() if len(finite) else np.inf):9.3f}mm"
        print(row)

    print(f"\n=== candidates = subsampled to {args.budget} ===")
    print(hdr)
    print("-" * len(hdr))
    for r in RADII:
        d = np.array(near_sub[r]); c = np.minimum(np.array(counts[r]), args.budget)
        m = np.array(miss[r])
        finite = d[np.isfinite(d)]
        row = (f"{r:6.1f}m{c.mean():11.0f}{np.percentile(c, 95):11.0f}"
               f"{m.mean() * 100:7.2f}%")
        for t in TOLERANCES:
            row += f"{(d <= t).mean() * 100:9.2f}%"
        row += f"{(finite.max() if len(finite) else np.inf):9.3f}mm"
        print(row)

    print("\nMISS% = truth lies OUTSIDE the ball -- the ranker can never recover these.")
    print("rec<=X = truth within X mm of the nearest candidate (the achievable ceiling).")


if __name__ == "__main__":
    main()
