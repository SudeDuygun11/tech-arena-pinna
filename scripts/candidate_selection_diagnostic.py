"""Route (b): does INFORMED selection inside the ball beat random subsampling?

Established already: a ~10mm ball around the registration seed contains truth
for 99.4% of anchors and reaches 97.5% recall@0.5mm -- but it holds ~1400
vertices, and RANDOM subsampling to 128 collapses recall@0.5mm to ~24%.

So the question is not "which ball" but "which 128 of the 1400". Random is the
worst possible answer because it clusters and leaves gaps. This compares:

  random            -- baseline, what was measured before
  fps               -- farthest-point sampling; maximises spatial coverage,
                       directly optimising "truth is near SOME candidate"
  curvature_top     -- highest |curvature| vertices (the teammate's extrema
                       instinct, applied inside the ball rather than as a
                       whole-mesh filter)
  curvature_weighted-- random draw weighted by |curvature|
  hybrid            -- half fps + half curvature_top

Reported per contour as well as overall, because 3 of 4 contours track
curvature ridges (0.81-1.08mm) while superior_antihelix does not (2.28mm) --
curvature-based selection should help the first three and hurt the fourth.

Uses the raw-registration seed: the cascade diagnostic showed seed choice
barely matters at the radii that give low miss rates, and raw needs no
trained model, so this runs fast.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, contour_of
from src.foundations.dataset import Dataset
from src.correction.patch_features import _local_curvature, crop_submesh
from src.registration.registration import _farthest_point_sample, compute_raw_full
from src.foundations.splits import k_fold_subject_split
from src.registration.template import build_template

TOLERANCES = [0.25, 0.50, 1.00]
STRATEGIES = ["random", "fps", "curvature_top", "curvature_weighted", "hybrid"]


def select(strategy: str, pts: np.ndarray, curv: np.ndarray, budget: int,
            rng: np.random.Generator) -> np.ndarray:
    """Return indices of the chosen candidates."""
    n = len(pts)
    if n <= budget:
        return np.arange(n)
    if strategy == "random":
        return rng.choice(n, budget, replace=False)
    if strategy == "fps":
        return _farthest_point_sample(pts, budget, seed=int(rng.integers(1 << 30)))
    if strategy == "curvature_top":
        return np.argsort(curv)[::-1][:budget]
    if strategy == "curvature_weighted":
        w = curv - curv.min() + 1e-9
        return rng.choice(n, budget, replace=False, p=w / w.sum())
    if strategy == "hybrid":
        half = budget // 2
        top = np.argsort(curv)[::-1][:half]
        rest = np.setdiff1d(np.arange(n), top, assume_unique=False)
        fps_idx = rest[_farthest_point_sample(pts[rest], budget - half,
                                               seed=int(rng.integers(1 << 30)))]
        return np.concatenate([top, fps_idx])
    raise ValueError(strategy)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=24)
    p.add_argument("--k-references", type=int, default=7)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--budgets", default="64,128,256")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    ids = ds.subject_ids[:args.n_subjects]
    train_ids, test_ids = next(iter(k_fold_subject_split(ids, 2, args.seed)))
    print(f"template from {len(train_ids)}; measuring on {len(test_ids)}  "
          f"radius={args.radius}mm")
    template = build_template(ds, train_ids, k_references=args.k_references)

    # dists[budget][strategy] -> list of truth->nearest-candidate distances
    dists = {b: {s: [] for s in STRATEGIES} for b in budgets}
    per_contour = {b: {s: {} for s in STRATEGIES} for b in budgets}
    full_ball = []
    rng = np.random.default_rng(args.seed)
    n_ears = 0

    for ex in iter_landmarks_only(ds, test_ids):
        mesh = load_canonical_mesh(ds, ex.subject_id, ex.side)
        raw = compute_raw_full(template, mesh.vertices, method="tps")
        local = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)
        V = np.asarray(local.vertices)
        curv_all = np.abs(_local_curvature(local, np.arange(len(V))))
        n_ears += 1

        for a in ANCHOR_INDICES:
            s, t = raw[a], ex.landmarks[a]
            inside = np.linalg.norm(V - s[None, :], axis=1) <= args.radius
            if not inside.any():
                continue
            pts, cv = V[inside], curv_all[inside]
            dt_all = np.linalg.norm(pts - t[None, :], axis=1)
            full_ball.append(float(dt_all.min()))
            cname = contour_of(a)
            for b in budgets:
                for st in STRATEGIES:
                    idx = select(st, pts, cv, b, rng)
                    d = float(dt_all[idx].min())
                    dists[b][st].append(d)
                    per_contour[b][st].setdefault(cname, []).append(d)

    fb = np.array(full_ball)
    print(f"\nears: {n_ears}   anchors: {len(fb)}")
    print(f"\nCEILING (all ~{int(np.mean([1]) * 0) or ''}vertices in ball, no subsampling):")
    print(f"  recall<=0.25 {(fb <= 0.25).mean() * 100:.2f}%   "
          f"<=0.50 {(fb <= 0.50).mean() * 100:.2f}%   "
          f"<=1.00 {(fb <= 1.00).mean() * 100:.2f}%   mean {fb.mean():.3f}mm")

    for b in budgets:
        print(f"\n{'=' * 70}\n=== budget = {b} candidates ===")
        hdr = f"{'strategy':22s}" + "".join(f"rec<={t:.2f}".rjust(11) for t in TOLERANCES) + f"{'mean':>10s}"
        print(hdr); print("-" * len(hdr))
        for st in STRATEGIES:
            d = np.array(dists[b][st])
            row = f"{st:22s}"
            for t in TOLERANCES:
                row += f"{(d <= t).mean() * 100:10.2f}%"
            row += f"{d.mean():9.3f}mm"
            print(row)

    print(f"\n{'=' * 70}\n=== per-contour recall<=0.50mm (budget {budgets[-1]}) ===")
    cnames = sorted(per_contour[budgets[-1]][STRATEGIES[0]].keys())
    hdr = f"{'strategy':22s}" + "".join(c[:16].rjust(18) for c in cnames)
    print(hdr); print("-" * len(hdr))
    for st in STRATEGIES:
        row = f"{st:22s}"
        for c in cnames:
            d = np.array(per_contour[budgets[-1]][st][c])
            row += f"{(d <= 0.50).mean() * 100:17.2f}%"
        print(row)


if __name__ == "__main__":
    main()
