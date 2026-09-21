"""Noise-robustness curve for anchor-pinned ICP: ground-truth anchors +
synthetic isotropic Gaussian jitter, swept from 0 (oracle) up past our real
~1.4mm anchor error.

Complements tier1_anchor_pinned_realpred.py (real saved predictions, one
noise level, ~1.3-1.4mm, structured/correlated error) with a controlled sweep
using independent per-anchor noise. Comparing the two tells us whether the
mechanism's benefit depends on error STRUCTURE (real predictions err in
anatomically consistent ways -- e.g. a whole contour drifting together) or
just magnitude (this test, iid per anchor). If synthetic jitter at ~1.4mm
matches the real-prediction result, magnitude is what matters and the effect
is robust; if it diverges, structure matters and the real-prediction number
is the one to trust.
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
from src.foundations.geometry import kabsch, apply_rigid
from src.pipeline import predict_canonical
from src.registration import registration as R
from src.registration import template as T

_PIN_ANCHORS = None


def rigid_icp_anchor_pinned(source, target, n_iters=R.RIGID_ICP_ITERS):
    Rm, t = np.eye(3), np.zeros(3)
    from scipy.spatial import cKDTree
    tree = cKDTree(target)
    warped = source.copy()
    anchor_src = source[ANCHOR_INDICES] if _PIN_ANCHORS is not None else None
    for _ in range(n_iters):
        dist, idx = tree.query(warped)
        keep = np.ones(len(dist), dtype=bool) if R.TRIM_FRACTION <= 0 else \
               dist <= np.quantile(dist, 1 - R.TRIM_FRACTION)
        src_pts, tgt_pts = source[keep], target[idx[keep]]
        if anchor_src is not None:
            src_pts = np.vstack([src_pts] + [anchor_src] * 3)
            tgt_pts = np.vstack([tgt_pts] + [_PIN_ANCHORS] * 3)
        Rm, t = kabsch(src_pts, tgt_pts)
        warped = apply_rigid(source, Rm, t)
    return Rm, t, warped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--k-references", type=int, default=11)
    ap.add_argument("--noise-mm", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.4, 2.0, 3.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    train_ids = ds.subject_ids[:args.n_train]
    test_ids = ds.subject_ids[args.n_train:args.n_train + args.n_test]

    print(f"template pool {len(train_ids)}, test {len(test_ids)} "
          f"(k_references={args.k_references})", flush=True)
    cache = [(e.subject_id, e.side, np.asarray(e.landmarks, float),
              load_canonical_mesh(ds, e.subject_id, e.side))
             for e in iter_landmarks_only(ds, test_ids)]
    print(f"  {len(cache)} ears cached\n", flush=True)

    template = T.build_template(ds, train_ids, k_references=args.k_references)
    non_anchor = np.ones(85, dtype=bool)
    non_anchor[ANCHOR_INDICES] = False
    rng = np.random.default_rng(args.seed)

    def score(noise_mm):
        global _PIN_ANCHORS
        orig = R._rigid_icp
        R._rigid_icp = rigid_icp_anchor_pinned
        try:
            errs = []
            for _, _, L, mesh in cache:
                if noise_mm is None:
                    _PIN_ANCHORS = None
                else:
                    jitter = rng.normal(scale=noise_mm, size=(len(ANCHOR_INDICES), 3))
                    _PIN_ANCHORS = L[ANCHOR_INDICES] + jitter
                pred = predict_canonical(template, None, mesh)
                errs.append(np.linalg.norm(pred[non_anchor] - L[non_anchor], axis=1).mean())
            return float(np.mean(errs))
        finally:
            R._rigid_icp = orig
            _PIN_ANCHORS = None

    blind = score(None)
    print(f"blind ICP (no pinning):  {blind:.4f} mm  (non-anchor points)\n", flush=True)

    print(f"{'jitter (mm)':>12s} {'error':>9s} {'delta':>9s}")
    for nm in args.noise_mm:
        s = score(nm)
        print(f"{nm:12.1f} {s:9.4f} {s-blind:+9.4f}", flush=True)

    print("\n0.0 = oracle ceiling. ~1.3-1.4mm should be comparable to the real-prediction")
    print("test (tier1_anchor_pinned_realpred.py). Where the curve crosses back above")
    print("the blind-ICP baseline (if it does) is the noise level past which pinning hurts.")


if __name__ == "__main__":
    main()
