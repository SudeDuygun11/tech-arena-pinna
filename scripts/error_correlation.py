"""Measure the SPATIAL CORRELATION of our real prediction errors.

correlated_noise_test.py established that a shape prior's ability to denoise
depends almost entirely on the correlation length of the error, and that the
dependence is NON-MONOTONIC (input 1.48mm, 45 modes):

    independent      -> 0.706mm   filtered well (prior has only 45 dof)
    2mm              -> 0.855mm
    5mm              -> 1.116mm   WORST
    10mm             -> 1.130mm   WORST
    fully coherent   -> 0.377mm   absorbed by Procrustes alignment

So which regime we are in decides whether the prior is worth anything. This
measures it directly from saved predictions instead of assuming.

Three things are reported:

  1. EMPIRICAL CORRELATION vs landmark separation -- the actual curve, fitted
     to exp(-d^2/2L^2) to extract L.
  2. RIGID vs RESIDUAL split -- how much of the error is a global rigid shift
     (which Procrustes removes for free) versus genuine local deformation.
  3. HOW WELL THE SYNTHETIC MODEL MATCHES -- compares the measured curve to the
     Gaussian random field used in correlated_noise_test.py. If they diverge,
     those projections do not transfer.

Input: an .npz from `cross_validate.py --save-predictions`.
Mesh-free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.geometry import apply_rigid, kabsch

BINS = [0, 2, 4, 6, 8, 12, 16, 22, 30, 45, 80]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help=".npz from --save-predictions")
    args = p.parse_args()

    d = np.load(args.predictions, allow_pickle=True)
    pred, truth = d["pred"], d["truth"]
    n_ears = len(pred)
    print(f"{n_ears} ears loaded from {args.predictions}")

    raw_err = np.linalg.norm(pred - truth, axis=2)
    print(f"raw error: mean {raw_err.mean():.4f}mm  p95 {np.percentile(raw_err, 95):.4f}mm\n")

    # ---- 2. how much of the error is a global rigid shift?
    resid_vecs, rigid_removed = [], []
    for i in range(n_ears):
        R, t = kabsch(pred[i], truth[i])          # best rigid alignment of pred onto truth
        aligned = apply_rigid(pred[i], R, t)
        resid_vecs.append(aligned - truth[i])
        rigid_removed.append(raw_err[i].mean() - np.linalg.norm(aligned - truth[i], axis=1).mean())
    resid_vecs = np.array(resid_vecs)
    resid_err = np.linalg.norm(resid_vecs, axis=2)

    print("=== rigid vs residual decomposition ===")
    print(f"  raw error                        {raw_err.mean():.4f}mm")
    print(f"  after removing best rigid fit    {resid_err.mean():.4f}mm")
    print(f"  -> rigid component               {np.mean(rigid_removed):.4f}mm "
          f"({100 * np.mean(rigid_removed) / raw_err.mean():.1f}% of error)")
    print("  (the rigid part is absorbed by the prior's Procrustes step for free;")
    print("   only the residual is what the prior must actually filter)\n")

    # ---- 1. correlation of the residual error vs landmark separation
    mean_shape = truth.mean(axis=0)
    n_pts = mean_shape.shape[0]
    iu = np.triu_indices(n_pts, k=1)
    sep = np.linalg.norm(mean_shape[iu[0]] - mean_shape[iu[1]], axis=1)

    # normalised error vectors, so correlation is about direction agreement
    v = resid_vecs - resid_vecs.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(v, axis=2, keepdims=True)
    vn = v / np.maximum(norms, 1e-9)
    # per-pair mean cosine similarity across ears
    cos = np.einsum("eik,ejk->eij", vn, vn).mean(axis=0)
    pair_corr = cos[iu]

    print("=== error correlation vs landmark separation ===")
    print(f"{'separation':>14s}{'n pairs':>10s}{'mean corr':>12s}")
    print("-" * 36)
    centres, values = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (sep >= lo) & (sep < hi)
        if m.sum() < 5:
            continue
        c = pair_corr[m].mean()
        centres.append(0.5 * (lo + hi)); values.append(c)
        print(f"{f'{lo}-{hi}mm':>14s}{m.sum():10d}{c:+12.3f}")

    centres, values = np.array(centres), np.array(values)
    pos = values > 0.01
    if pos.sum() >= 3:
        # fit corr = exp(-d^2 / 2L^2)  ->  log(corr) = -d^2 / 2L^2
        slope = np.polyfit(centres[pos] ** 2, np.log(values[pos]), 1)[0]
        L = np.sqrt(-1.0 / (2 * slope)) if slope < 0 else np.inf
        print(f"\n  fitted correlation length L ~ {L:.1f}mm")
        if L < 3:
            verdict = "NEAR-INDEPENDENT -> prior should denoise well (~0.71-0.86mm regime)"
        elif L < 15:
            verdict = "WORST BAND (5-10mm) -> prior filters poorly (~1.13mm regime)"
        else:
            verdict = "LARGELY COHERENT -> mostly absorbed by Procrustes"
        print(f"  -> {verdict}")
    else:
        print("\n  correlation is negligible at all separations -> effectively independent")

    print("\n=== does the synthetic Gaussian-random-field model match? ===")
    if pos.sum() >= 3 and np.isfinite(L):
        model = np.exp(-centres ** 2 / (2 * L ** 2))
        print(f"{'separation':>14s}{'measured':>11s}{'GRF model':>12s}{'diff':>9s}")
        print("-" * 46)
        for c, meas, mod in zip(centres, values, model):
            print(f"{c:13.0f}m{meas:+11.3f}{mod:+12.3f}{meas - mod:+9.3f}")
        print("\n  Large diffs mean the projections in correlated_noise_test.py do not")
        print("  transfer, and its predicted numbers should not be relied on.")


if __name__ == "__main__":
    main()
