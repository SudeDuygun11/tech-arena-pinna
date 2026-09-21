"""Candidate-set measurement for THREE possible seed sources.

The raw-registration seed was measured at 3.910mm mean / 7.714mm p95, which
forces a ~10mm candidate ball (0.56% miss) holding ~1400 vertices -- and
random-subsampling that to 128 collapses recall from 97.5% to 28.3%.

This script asks whether seeding on a LATER pipeline stage fixes that, by
measuring the same quantities for:

    (1) raw registration            -- the baseline already measured
    (2) after Stage-1b correction   -- corrected anchors
    (3) after blend + polish        -- final pipeline output

A better seed means a smaller ball, fewer candidates, and less need for
informed subsampling. Leakage-free throughout: template and both networks are
fit on subjects disjoint from those measured.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correction.anchor_model import predict_correction, train_model
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, ANCHOR_POSITION, N_LANDMARKS
from src.foundations.dataset import Dataset
from src.interpolation.interpolation import blend_correct
from src.correction.patch_features import crop_submesh, extract_patch
from src.registration.registration import compute_raw_full
from src.foundations.splits import k_fold_subject_split
from src.registration.template import build_template
from src.correction.training_data import generate_inner_cv_examples, generate_polish_examples

RADII = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
TOLERANCES = [0.25, 0.50, 1.00]


def report(name, seed_err, counts, miss, near_all, near_sub, budget):
    se = np.array(seed_err)
    print(f"\n{'=' * 78}\n=== SEED: {name} ===")
    print(f"  seed error: mean {se.mean():.3f}mm  median {np.median(se):.3f}mm  "
          f"p95 {np.percentile(se, 95):.3f}mm  max {se.max():.3f}mm")
    for label, near in (("ALL vertices in ball", near_all),
                         (f"subsampled to {budget}", near_sub)):
        hdr = (f"{'radius':>7s}{'n_vtx':>8s}{'MISS%':>8s}"
               + "".join(f"rec<={t:.2f}".rjust(10) for t in TOLERANCES) + f"{'worst':>10s}")
        print(f"\n  -- candidates = {label} --")
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for r in RADII:
            d = np.array(near[r]); c = np.array(counts[r]); m = np.array(miss[r])
            if label.startswith("subsampled"):
                c = np.minimum(c, budget)
            fin = d[np.isfinite(d)]
            row = f"{r:6.1f}m{c.mean():8.0f}{m.mean() * 100:7.2f}%"
            for t in TOLERANCES:
                row += f"{(d <= t).mean() * 100:9.2f}%"
            row += f"{(fin.max() if len(fin) else np.inf):9.3f}mm"
            print("  " + row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=24)
    p.add_argument("--k-references", type=int, default=7)
    p.add_argument("--inner-folds", type=int, default=2)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--capacity", default="large")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-seed", type=int, default=0)
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    ids = ds.subject_ids[:args.n_subjects]
    train_ids, test_ids = next(iter(k_fold_subject_split(ids, 2, args.seed)))
    print(f"train {len(train_ids)} / measure {len(test_ids)} subjects "
          f"(capacity={args.capacity}, epochs={args.epochs})")

    t0 = time.time()
    ex_anchor = generate_inner_cv_examples(ds, train_ids, k_inner=args.inner_folds,
                                            k_references=args.k_references, seed=args.seed,
                                            verbose=True)
    model = train_model(ex_anchor, n_epochs=args.epochs, capacity=args.capacity,
                         verbose=False, torch_seed=args.torch_seed)
    print(f"correction model trained ({time.time() - t0:.0f}s)")

    template = build_template(ds, train_ids, k_references=args.k_references)
    t0 = time.time()
    ex_polish = generate_polish_examples(ds, template, model, train_ids, seed=args.seed)
    polish = train_model(ex_polish, n_epochs=args.epochs, n_classes=N_LANDMARKS,
                          capacity=args.capacity, verbose=False, torch_seed=args.torch_seed)
    print(f"polish model trained ({time.time() - t0:.0f}s)\n")

    stages = ["raw registration", "after Stage-1b correction", "after blend + polish"]
    acc = {s: dict(seed_err=[], counts={r: [] for r in RADII}, miss={r: [] for r in RADII},
                    near_all={r: [] for r in RADII}, near_sub={r: [] for r in RADII})
           for s in stages}
    rng = np.random.default_rng(args.seed)

    for ex in iter_landmarks_only(ds, test_ids):
        mesh = load_canonical_mesh(ds, ex.subject_id, ex.side)
        raw = compute_raw_full(template, mesh.vertices, method="tps")
        local = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)
        V = np.asarray(local.vertices)

        corrected = {}
        for a in ANCHOR_INDICES:
            patch = extract_patch(local, raw[a])
            corrected[a] = raw[a] + predict_correction(model, patch, ANCHOR_POSITION[a])
        interp = blend_correct(raw, corrected, mesh.vertices)
        final = interp.copy()
        for i in range(N_LANDMARKS):
            patch = extract_patch(local, interp[i])
            final[i] = interp[i] + predict_correction(polish, patch, i)

        seeds = {stages[0]: {a: raw[a] for a in ANCHOR_INDICES},
                 stages[1]: corrected,
                 stages[2]: {a: final[a] for a in ANCHOR_INDICES}}

        for sname, sdict in seeds.items():
            A = acc[sname]
            for a in ANCHOR_INDICES:
                s, t = sdict[a], ex.landmarks[a]
                A["seed_err"].append(float(np.linalg.norm(s - t)))
                dfs = np.linalg.norm(V - s[None, :], axis=1)
                for r in RADII:
                    inside = dfs <= r
                    A["counts"][r].append(int(inside.sum()))
                    A["miss"][r].append(bool(np.linalg.norm(s - t) > r))
                    if not inside.any():
                        A["near_all"][r].append(np.inf); A["near_sub"][r].append(np.inf); continue
                    cand = V[inside]
                    dt = np.linalg.norm(cand - t[None, :], axis=1)
                    A["near_all"][r].append(float(dt.min()))
                    if len(cand) > args.budget:
                        sel = rng.choice(len(cand), args.budget, replace=False)
                        A["near_sub"][r].append(float(dt[sel].min()))
                    else:
                        A["near_sub"][r].append(float(dt.min()))

    for s in stages:
        A = acc[s]
        report(s, A["seed_err"], A["counts"], A["miss"], A["near_all"], A["near_sub"], args.budget)

    print("\nMISS% = truth outside the ball -- unrecoverable by any ranker.")
    print("rec<=X = truth within X mm of nearest candidate = the achievable ceiling.")


if __name__ == "__main__":
    main()
