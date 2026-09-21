"""How much of our 1.59mm is INTERPOLATION error rather than anchor error?

Only 15 of 85 points are predicted; the other 70 come from blend_correct. The
project record shows blend_correct evaluated with GROUND-TRUTH anchors still
gives ~0.9-1.6mm per contour. If that holds, then even PERFECT anchors leave
roughly (70 x 1.2)/85 ~ 1.0mm overall -- i.e. interpolation, not the anchor
model, is what stands between us and a sub-1mm target.

This script asks whether the PCA shape prior interpolates better than blending,
and how the answer changes as MORE points are observed:

    observe N points (ground truth) -> fit the 85-point prior -> reconstruct
    all 85 -> measure error on the UNOBSERVED points

Deliberately mesh-free (landmarks only) so it is safe to run alongside a
full-scale job.

Two observation regimes, because the optimistic one alone would mislead:
  PERFECT  observed points are exact -> the interpolation ceiling
  NOISY    observed points get realistic anchor error added -> what a real
           pipeline would actually see
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only
from src.foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS, N_LANDMARKS
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid, kabsch
from src.foundations.splits import k_fold_subject_split

N_MODES = 45
RIDGE = 0.05          # matches the validated prior strength
OBSERVED_COUNTS = [15, 20, 25, 30, 40, 55, 70, 85]
NOISE_LEVELS = [0.0, 1.0, 1.6]   # mm, added to observed points


def procrustes_align(shapes, n_iter=5):
    aligned = shapes.copy()
    mean = aligned[0]
    for _ in range(n_iter):
        for i in range(len(aligned)):
            R, t = kabsch(aligned[i], mean)
            aligned[i] = apply_rigid(aligned[i], R, t)
        mean = aligned.mean(axis=0)
    return aligned, mean


def choose_observed(n: int) -> np.ndarray:
    """Which points to observe: the 15 anatomical anchors first (they are what a
    human actually judges), then fill in evenly along each contour."""
    chosen = list(ANCHOR_INDICES)
    if n <= len(chosen):
        return np.array(sorted(chosen[:n]))
    remaining = n - len(chosen)
    pool = []
    for spec in CONTOUR_SPECS.values():
        s, e = spec["range"]
        pool.extend([i for i in range(s, e) if i not in chosen])
    pool = np.array(sorted(pool))
    step = max(1, len(pool) // remaining)
    chosen.extend(pool[::step][:remaining].tolist())
    return np.array(sorted(set(chosen)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=200)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    shapes = np.array([e.landmarks for e in
                        iter_landmarks_only(ds, ds.subject_ids[:args.n_subjects])])
    print(f"{len(shapes)} ears, mesh-free\n")

    rng = np.random.default_rng(args.seed)
    idx_all = np.arange(len(shapes))
    # results[noise][n_obs] -> (unobserved_err, overall_err)
    res = {nz: {n: ([], []) for n in OBSERVED_COUNTS} for nz in NOISE_LEVELS}

    for tr, te in k_fold_subject_split(list(idx_all), args.folds, args.seed):
        tr = np.array([int(i) for i in tr]); te = np.array([int(i) for i in te])
        aligned, mean_shape = procrustes_align(shapes[tr].copy())
        X = aligned.reshape(len(tr), -1)
        mu = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        P = Vt[:N_MODES]
        evals = (S[:N_MODES] ** 2) / (len(X) - 1)
        Linv = np.diag(1.0 / np.maximum(evals, 1e-9))

        for i in te:
            truth = shapes[i]
            R, t = kabsch(truth, mean_shape)
            truth_a = apply_rigid(truth, R, t)
            for n_obs in OBSERVED_COUNTS:
                obs_pts = choose_observed(n_obs)
                unobs = np.setdiff1d(np.arange(N_LANDMARKS), obs_pts)
                rows = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in obs_pts])
                Po = P[:, rows]
                A = Po @ Po.T + RIDGE * Linv
                for nz in NOISE_LEVELS:
                    obs = truth_a[obs_pts]
                    if nz > 0:
                        obs = obs + rng.normal(scale=nz / np.sqrt(3), size=obs.shape)
                    b = np.linalg.solve(A, Po @ (obs.reshape(-1) - mu[rows]))
                    rec = (mu + P.T @ b).reshape(-1, 3)
                    e = np.linalg.norm(rec - truth_a, axis=1)
                    res[nz][n_obs][0].append(e[unobs].mean() if len(unobs) else 0.0)
                    res[nz][n_obs][1].append(e.mean())

    for nz in NOISE_LEVELS:
        label = ("PERFECT observations (interpolation ceiling)" if nz == 0
                 else f"NOISY observations (+{nz:.1f}mm error, realistic)")
        print("=" * 72)
        print(f"=== {label} ===")
        print(f"{'observed':>10s}{'unobserved':>14s}{'ALL 85 pts':>14s}{'note':>22s}")
        print("-" * 60)
        for n in OBSERVED_COUNTS:
            un = np.mean(res[nz][n][0]); ov = np.mean(res[nz][n][1])
            note = "current design" if n == 15 else ("all observed" if n == 85 else "")
            print(f"{n:10d}{un:13.3f}m{ov:13.3f}m{note:>22s}")
        print()

    print("Reference points:")
    print("  current full pipeline                 1.5931mm")
    print("  blend_correct with GT anchors  ~0.9-1.6mm per contour (project record)")
    print("\nRead the PERFECT/observed=15 row against blend_correct's GT-anchor number:")
    print("that is the like-for-like comparison of the two interpolation schemes.")
    print("The NOISY tables show whether the advantage survives real anchor error.")


if __name__ == "__main__":
    main()
