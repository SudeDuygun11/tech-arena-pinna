"""Ceiling test: does PINNING anchor correspondences during rigid ICP (instead
of leaving them to nearest-neighbor matching) improve the NON-anchor points?

WHY
---
The Jain et al. 2025 pinna-registration paper solves a joint energy where
landmark correspondences constrain the deformation directly, rather than
running ICP blind and reading landmarks off the result afterward (our current
order). The full version is expensive to build. This is the cheap oracle
version: use GROUND-TRUTH anchor positions (an upper bound -- our own anchor
predictions are noisier) as extra, hard-weighted correspondences during rigid
ICP, and see whether that changes the resulting pose enough to help the OTHER
70 points. If even the oracle version doesn't help, a prediction-based version
(which would need real anchor predictions, i.e. an actual training run) isn't
worth building. If it does, that's the evidence needed to justify one.

Anchor points are excluded from the scored error here -- pinning them to
ground truth would trivially make them near-perfect, which tells us nothing
about whether the MECHANISM (a better rigid pose) helps.

METHOD
------
source[ANCHOR_INDICES] (the reference's own anchor positions, since
control_points has landmarks as its first 85 rows) are paired directly with
this ear's ground-truth anchor positions and appended to the correspondence
set every ICP iteration, so the alternating Kabsch step aligns TOWARD the
correct pose from the first iteration rather than converging there slowly (or
not at all) via nearest-neighbor alone.
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

_ORACLE_ANCHORS = None   # set per-ear before each predict_canonical call


def rigid_icp_anchor_pinned(source, target, n_iters=R.RIGID_ICP_ITERS):
    """Drop-in replacement for R._rigid_icp. If _ORACLE_ANCHORS is set, the
    reference's own anchor rows are paired with it and added to every
    iteration's correspondence set (not just used to initialise)."""
    Rm, t = np.eye(3), np.zeros(3)
    from scipy.spatial import cKDTree
    tree = cKDTree(target)
    warped = source.copy()

    anchor_src = source[ANCHOR_INDICES] if _ORACLE_ANCHORS is not None else None

    for _ in range(n_iters):
        dist, idx = tree.query(warped)
        keep = np.ones(len(dist), dtype=bool) if R.TRIM_FRACTION <= 0 else \
               dist <= np.quantile(dist, 1 - R.TRIM_FRACTION)
        src_pts, tgt_pts = source[keep], target[idx[keep]]
        if anchor_src is not None:
            # append pinned correspondences; duplicated 3x so the least-squares
            # Kabsch fit can't just average them away against ~430 other points
            src_pts = np.vstack([src_pts] + [anchor_src] * 3)
            tgt_pts = np.vstack([tgt_pts] + [_ORACLE_ANCHORS] * 3)
        Rm, t = kabsch(src_pts, tgt_pts)
        warped = apply_rigid(source, Rm, t)
    return Rm, t, warped


def score(template, cache, pinned: bool):
    global _ORACLE_ANCHORS
    orig = R._rigid_icp
    R._rigid_icp = rigid_icp_anchor_pinned
    non_anchor_mask = np.ones(85, dtype=bool)
    non_anchor_mask[ANCHOR_INDICES] = False
    try:
        errs = []
        for _, _, L, mesh in cache:
            _ORACLE_ANCHORS = L[ANCHOR_INDICES] if pinned else None
            pred = predict_canonical(template, None, mesh)
            errs.append(np.linalg.norm(pred[non_anchor_mask] - L[non_anchor_mask], axis=1).mean())
        return float(np.mean(errs))
    finally:
        R._rigid_icp = orig
        _ORACLE_ANCHORS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--k-references", type=int, default=11)
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

    blind = score(template, cache, pinned=False)
    print(f"blind ICP (current):        {blind:.4f} mm  (non-anchor points only)", flush=True)

    pinned = score(template, cache, pinned=True)
    print(f"anchor-pinned ICP (oracle): {pinned:.4f} mm  ({pinned-blind:+.4f})", flush=True)

    print("\nThis is a CEILING using ground-truth anchors, not real predictions.")
    print("If it doesn't beat blind ICP here, a prediction-based version won't either.")
    print("If it does, the gap is the most this mechanism could ever be worth.")


if __name__ == "__main__":
    main()
