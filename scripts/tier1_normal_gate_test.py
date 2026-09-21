"""Does a distance+normal-agreement correspondence gate beat distance-only
(current: TRIM_FRACTION=0, i.e. no gate at all)?

WHY
---
The Jain et al. 2025 pinna-registration paper (DAGA 2025) rejects ICP
correspondences using BOTH a distance criterion AND surface-normal agreement,
not a fixed trim percentage. We already found that our old distance-only
15% trim was actively harmful (0.15 -> 0.0 measured -0.2165mm registration-
only) -- but "no gate at all" and "a smarter gate" are different hypotheses.
This tests whether adding normal agreement on top of TRIM_FRACTION=0 helps,
without touching the production _rigid_icp yet.

METHOD
------
Self-contained: normals for both point sets are estimated locally via PCA
over k nearest neighbours (no face/mesh connectivity needed, since neither
`reference.control_points` nor the cropped target vertices carry it through
to this call). Source-point normals are computed once and rotated by the
current R each iteration (normals transform by rotation only); target
normals are computed once since target is fixed.

Registration-only scoring, same harness as tier1_registration_sweep.py --
~10x cheaper than a training run, and this machine is memory-constrained
right now so a light test is deliberate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.dataset import Dataset
from src.foundations.geometry import kabsch, apply_rigid
from src.pipeline import predict_canonical
from src.registration import registration as R
from src.registration import template as T


def local_normals(points: np.ndarray, k: int = 12) -> np.ndarray:
    """Per-point normal via PCA over k nearest neighbours: the eigenvector of
    smallest variance. Sign is arbitrary (fine here -- only used for |dot|
    agreement, not orientation).

    Vectorised: numpy's SVD batches over leading dimensions, so all N
    neighbourhoods are decomposed in one call instead of a Python loop calling
    SVD once per point. The first version of this script did the latter and
    was too slow to finish even a 3-subject smoke test in 400s -- worth noting
    since it's an easy trap to fall into with "per-point geometry" code."""
    k = min(k, len(points))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)                    # (N, k)
    nb = points[idx]                                    # (N, k, 3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    # smallest-singular-vector direction per point, batched
    _, _, Vt = np.linalg.svd(nb, full_matrices=False)    # Vt: (N, 3, 3)
    out = Vt[:, -1, :]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


def rigid_icp_normal_gated(source, target, n_iters, trim_fraction, normal_thresh):
    """Same alternating Kabsch loop as R._rigid_icp, plus an optional
    normal-agreement filter on top of the (possibly zero) distance trim.
    normal_thresh=None reproduces the current production behaviour exactly."""
    src_n = local_normals(source)
    tgt_n = local_normals(target) if normal_thresh is not None else None
    Rm, t = np.eye(3), np.zeros(3)
    tree = cKDTree(target)
    warped = source.copy()
    for _ in range(n_iters):
        dist, idx = tree.query(warped)
        keep = dist <= np.quantile(dist, 1 - trim_fraction) if trim_fraction > 0 else \
               np.ones(len(dist), dtype=bool)
        if normal_thresh is not None:
            warped_n = src_n @ Rm.T          # normals rotate with the current pose
            agree = np.abs(np.einsum("ij,ij->i", warped_n, tgt_n[idx]))
            keep = keep & (agree > normal_thresh)
            if keep.sum() < 10:              # degenerate: gate too strict, skip it
                keep = dist <= np.quantile(dist, 1 - trim_fraction) if trim_fraction > 0 else \
                       np.ones(len(dist), dtype=bool)
        Rm, t = kabsch(source[keep], target[idx[keep]])
        warped = apply_rigid(source, Rm, t)
    return Rm, t, warped


def score_with_gate(template, cache, normal_thresh):
    """Re-implements registration scoring with the gated rigid ICP swapped in
    for R._rigid_icp, via a temporary monkeypatch (restored after)."""
    orig = R._rigid_icp

    def patched(source, target, n_iters=R.RIGID_ICP_ITERS):
        return rigid_icp_normal_gated(source, target, n_iters, R.TRIM_FRACTION, normal_thresh)

    R._rigid_icp = patched
    try:
        errs = []
        for _, _, L, mesh in cache:
            pred = predict_canonical(template, None, mesh)
            errs.append(np.linalg.norm(pred - L, axis=1).mean())
        return float(np.mean(errs))
    finally:
        R._rigid_icp = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--k-references", type=int, default=11)
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    train_ids = ds.subject_ids[:args.n_train]
    test_ids = ds.subject_ids[args.n_train:args.n_train + args.n_test]

    print(f"template pool {len(train_ids)}, test {len(test_ids)}  "
          f"(k_references={args.k_references})", flush=True)
    cache = [(e.subject_id, e.side, np.asarray(e.landmarks, float),
              load_canonical_mesh(ds, e.subject_id, e.side))
             for e in iter_landmarks_only(ds, test_ids)]
    print(f"  {len(cache)} ears cached\n", flush=True)

    template = T.build_template(ds, train_ids, k_references=args.k_references)

    baseline = score_with_gate(template, cache, None)   # None = current prod behaviour
    print(f"baseline (no normal gate, current TRIM_FRACTION={R.TRIM_FRACTION}): "
          f"{baseline:.4f} mm\n", flush=True)

    for thresh in (0.3, 0.5, 0.7, 0.9):
        s = score_with_gate(template, cache, thresh)
        print(f"  normal_thresh={thresh:.1f}  -> {s:.4f} mm  ({s-baseline:+.4f})", flush=True)

    print("\nA threshold beating baseline by more than run-to-run noise (~0.05-0.1mm")
    print("at this scale) is worth a Tier 2 confirmation; anything within that band isn't.")


if __name__ == "__main__":
    main()
