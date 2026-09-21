"""How much does CORRELATED error destroy the shape prior's denoising?

interpolation_ceiling.py showed that observing all 85 points with +1.6mm
INDEPENDENT noise reconstructs to 0.710mm -- the prior averages the noise away.
But real pipeline errors are not independent: one registration seeds every
point, one model predicts them all, and neighbouring points read overlapping
patches. A coherent regional drift looks like plausible shape variation, so the
prior should pass it straight through.

This quantifies that. Noise is drawn from a Gaussian random field over the 85
landmarks with correlation length L:

    Cov(i, j) = sigma^2 * exp(-d(i,j)^2 / (2 L^2))

    L -> 0    independent per-point noise (the optimistic case already measured)
    L -> inf  every point shifts together (pure rigid drift)

Also sweeps the number of prior modes, because a lower-rank prior can follow
less correlated drift -- it should reject more error at the cost of rejecting
real variation too.

Mesh-free, so safe to run alongside a full-scale job.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid, kabsch
from src.foundations.splits import k_fold_subject_split

RIDGE = 0.05
NOISE_MM = 1.6
CORR_LENGTHS = [0.0, 2.0, 5.0, 10.0, 20.0, 1e6]   # mm
MODE_COUNTS = [10, 20, 30, 45]


def procrustes_align(shapes, n_iter=5):
    aligned = shapes.copy()
    mean = aligned[0]
    for _ in range(n_iter):
        for i in range(len(aligned)):
            R, t = kabsch(aligned[i], mean)
            aligned[i] = apply_rigid(aligned[i], R, t)
        mean = aligned.mean(axis=0)
    return aligned, mean


def correlated_noise(mean_shape, L, sigma, rng):
    """Gaussian random field over the landmarks with correlation length L."""
    n = len(mean_shape)
    if L <= 0:
        return rng.normal(scale=sigma / np.sqrt(3), size=(n, 3))
    d2 = ((mean_shape[:, None, :] - mean_shape[None, :, :]) ** 2).sum(-1)
    C = np.exp(-d2 / (2 * L ** 2)) + 1e-8 * np.eye(n)
    Lc = np.linalg.cholesky(C)
    return (Lc @ rng.normal(size=(n, 3))) * (sigma / np.sqrt(3))


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
    print(f"{len(shapes)} ears | observing all 85 points | input noise {NOISE_MM}mm\n")

    rng = np.random.default_rng(args.seed)
    res = {L: {k: [] for k in MODE_COUNTS} for L in CORR_LENGTHS}
    raw_in = {L: [] for L in CORR_LENGTHS}

    for tr, te in k_fold_subject_split(list(np.arange(len(shapes))), args.folds, args.seed):
        tr = np.array([int(i) for i in tr]); te = np.array([int(i) for i in te])
        aligned, mean_shape = procrustes_align(shapes[tr].copy())
        X = aligned.reshape(len(tr), -1)
        mu = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)

        for i in te:
            R, t = kabsch(shapes[i], mean_shape)
            truth = apply_rigid(shapes[i], R, t)
            for L in CORR_LENGTHS:
                nz = correlated_noise(mean_shape, L, NOISE_MM, rng)
                obs = truth + nz
                raw_in[L].append(np.linalg.norm(obs - truth, axis=1).mean())
                for k in MODE_COUNTS:
                    P = Vt[:k]
                    ev = (S[:k] ** 2) / (len(X) - 1)
                    A = P @ P.T + RIDGE * np.diag(1.0 / np.maximum(ev, 1e-9))
                    b = np.linalg.solve(A, P @ (obs.reshape(-1) - mu))
                    rec = (mu + P.T @ b).reshape(-1, 3)
                    res[L][k].append(np.linalg.norm(rec - truth, axis=1).mean())

    hdr = f"{'corr length':>13s}{'input err':>12s}" + "".join(f"{k} modes".rjust(11)
                                                               for k in MODE_COUNTS)
    print(hdr); print("-" * len(hdr))
    for L in CORR_LENGTHS:
        lbl = "independent" if L == 0 else ("fully coherent" if L > 1e5 else f"{L:.0f}mm")
        row = f"{lbl:>13s}{np.mean(raw_in[L]):11.3f}m"
        for k in MODE_COUNTS:
            row += f"{np.mean(res[L][k]):10.3f}m"
        print(row)

    print("\nInput err is the noise actually injected; the mode columns are what the")
    print("prior recovers. Denoising works where output << input.")
    print("Ear landmark span is ~75mm, so a correlation length of 10-20mm means")
    print("roughly a quarter of the ear drifting together -- the realistic regime")
    print("for registration drift.")


if __name__ == "__main__":
    main()
