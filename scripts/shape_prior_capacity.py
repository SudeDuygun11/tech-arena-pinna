"""Decision 1 test: is a PCA shape prior PRECISE enough to constrain our anchors?

The question research cannot answer for us. A shape prior only helps if it can
represent a real ear's anchor configuration MORE accurately than our current
predictions. Otherwise it acts as a ceiling, not a constraint:

  reconstruction error << 1.593mm  -> prior is sharper than our predictions;
                                      joint fitting can genuinely constrain them
  reconstruction error >= 1.593mm  -> prior is coarser than what we already do;
                                      pulling toward it would ADD error

Method: Procrustes-align true anchor configurations (rigid only -- rotation and
translation, NOT scale, so results stay in millimetres and real size variation
is modelled rather than normalised away), fit PCA on training subjects, then
reconstruct HELD-OUT subjects' TRUE configurations using k modes.

Mode count is chosen by cross-validated generalisation error rather than a
variance threshold, per Model-order selection in statistical shape models
(arXiv:1808.00309), which shows the usual 90-98%-variance heuristic is not
principled and that fewer modes generalise better on limited data.

Both the 15-anchor model (45 dims) and the full 85-point model (255 dims) are
reported: with 400 shape samples the former is comfortably determined, the
latter much less so, and the gap between them is itself informative.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only
from src.foundations.contours import ANCHOR_INDICES
from src.foundations.dataset import Dataset
from src.foundations.geometry import kabsch, apply_rigid
from src.foundations.splits import k_fold_subject_split

MODE_COUNTS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 45]


def procrustes_align(shapes: np.ndarray, n_iter: int = 5):
    """Rigid-only generalised Procrustes. Returns aligned shapes and the mean."""
    aligned = shapes.copy()
    mean = aligned[0]
    for _ in range(n_iter):
        for i in range(len(aligned)):
            R, t = kabsch(aligned[i], mean)
            aligned[i] = apply_rigid(aligned[i], R, t)
        mean = aligned.mean(axis=0)
    return aligned, mean


def pca_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt, (S ** 2) / (len(X) - 1)


def reconstruct(x: np.ndarray, mu: np.ndarray, Vt: np.ndarray, k: int):
    if k == 0:
        return mu.copy()
    B = Vt[:k]
    return mu + (x - mu) @ B.T @ B


def run(name, shapes, n_folds, seed):
    n_pts = shapes.shape[1]
    print(f"\n{'=' * 74}\n=== {name}: {len(shapes)} shapes x {n_pts} points "
          f"({n_pts * 3} dims) ===")
    idx = np.arange(len(shapes))
    errs = {k: [] for k in MODE_COUNTS if k <= min(n_pts * 3, len(shapes) - 1)}
    var_explained = None

    for tr, te in k_fold_subject_split(list(idx), n_folds, seed):
        tr = np.array([int(i) for i in tr]); te = np.array([int(i) for i in te])
        aligned_tr, mean_shape = procrustes_align(shapes[tr].copy())
        X = aligned_tr.reshape(len(tr), -1)
        mu, Vt, ev = pca_fit(X)
        if var_explained is None:
            var_explained = np.cumsum(ev) / ev.sum()

        for i in te:
            s = shapes[i]
            R, t = kabsch(s, mean_shape)            # align test shape to model frame
            sa = apply_rigid(s, R, t)
            for k in errs:
                rec = reconstruct(sa.reshape(-1), mu, Vt, k).reshape(n_pts, 3)
                errs[k].append(np.linalg.norm(rec - sa, axis=1).mean())

    print(f"{'modes':>7s}{'var explained':>16s}{'recon mean':>13s}{'median':>10s}{'p95':>10s}")
    print("-" * 56)
    best_k, best_v = None, np.inf
    for k in sorted(errs):
        e = np.array(errs[k])
        ve = var_explained[k - 1] * 100 if k <= len(var_explained) else 100.0
        print(f"{k:7d}{ve:15.1f}%{e.mean():12.3f}m{np.median(e):9.3f}m"
              f"{np.percentile(e, 95):9.3f}m")
        if e.mean() < best_v:
            best_v, best_k = e.mean(), k
    print(f"\n  best: {best_k} modes -> {best_v:.3f}mm mean reconstruction error")
    print(f"  vs current pipeline error 1.593mm  ->  ", end="")
    if best_v < 1.593 * 0.7:
        print("PRIOR IS SHARPER: can meaningfully constrain")
    elif best_v < 1.593:
        print("prior is slightly sharper: limited constraining power")
    else:
        print("PRIOR IS COARSER than our predictions: would ADD error")
    return best_k, best_v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=200)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    full, anchors = [], []
    for ex in iter_landmarks_only(ds, ds.subject_ids[:args.n_subjects]):
        full.append(ex.landmarks)
        anchors.append(ex.landmarks[ANCHOR_INDICES])
    full, anchors = np.array(full), np.array(anchors)
    print(f"loaded {len(full)} ears from {args.n_subjects} subjects")

    run("15-ANCHOR model", anchors, args.folds, args.seed)
    run("FULL 85-point model", full, args.folds, args.seed)

    print("\nNote: reconstruction here uses the TRUE configuration as input -- it is")
    print("the prior's best case. Fitting from noisy predictions can only be worse.")


if __name__ == "__main__":
    main()
