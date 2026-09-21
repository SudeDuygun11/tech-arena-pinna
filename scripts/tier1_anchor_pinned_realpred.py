"""Realistic version of the anchor-pinned ICP test: pin to our ACTUAL saved
anchor predictions (results/tier2_trim0_iters16_kref11.npz, the confirmed
1.3736mm run) instead of ground truth.

The oracle version (tier1_anchor_pinned_icp.py) measured a ceiling of
-0.40 to -0.44mm on non-anchor points using ground-truth anchors as the pin.
That upper-bounds the mechanism but says nothing about what real, noisy
predictions (currently ~1.48mm anchor error) would deliver. This answers that
without any new training: the predictions already exist.

LEAKAGE CARE
------------
The saved predictions cover all 400 ears (200 subjects) across 3 CV folds --
every subject appears as a test subject in exactly one fold. So the Tier 1
template pool here is built from a DISJOINT slice of subject_ids (the tail)
from the test ears drawn from the saved predictions (the head), guaranteeing
no subject is both a template-reference candidate and a scored test ear.
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
    ap.add_argument("--preds", default="results/tier2_trim0_iters16_kref11.npz")
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--k-references", type=int, default=11)
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))

    d = np.load(args.preds, allow_pickle=True)
    pred_all, truth_all = d["pred"], d["truth"]
    sids_all, sides_all = d["subject_id"], d["side"]

    # Template pool: the TAIL of subject_ids. Test ears: pulled from the saved
    # predictions but restricted to subjects in the HEAD, so the two never overlap.
    all_subj = ds.subject_ids
    train_ids = all_subj[-args.n_train:]
    head_subj = set(all_subj[:100])

    test_i = [i for i in range(len(sids_all)) if str(sids_all[i]) in head_subj][:args.n_test]
    print(f"template pool: {len(train_ids)} subjects (tail)  |  "
          f"{len(test_i)} test ears (head, from saved predictions)", flush=True)

    cache = []
    for i in test_i:
        sid, side = str(sids_all[i]), str(sides_all[i])
        mesh = load_canonical_mesh(ds, sid, side)
        cache.append((sid, side, truth_all[i], pred_all[i], mesh))
    print(f"  {len(cache)} ears cached\n", flush=True)

    template = T.build_template(ds, train_ids, k_references=args.k_references)

    non_anchor = np.ones(85, dtype=bool)
    non_anchor[ANCHOR_INDICES] = False

    def score(use_pred_pin: bool):
        global _PIN_ANCHORS
        orig = R._rigid_icp
        R._rigid_icp = rigid_icp_anchor_pinned
        try:
            errs = []
            for sid, side, truth, pred, mesh in cache:
                _PIN_ANCHORS = pred[ANCHOR_INDICES] if use_pred_pin else None
                out = predict_canonical(template, None, mesh)
                errs.append(np.linalg.norm(out[non_anchor] - truth[non_anchor], axis=1).mean())
            return float(np.mean(errs))
        finally:
            R._rigid_icp = orig
            _PIN_ANCHORS = None

    blind = score(False)
    print(f"blind ICP:                        {blind:.4f} mm  (non-anchor points)", flush=True)

    real_pin = score(True)
    print(f"pinned to REAL saved predictions: {real_pin:.4f} mm  ({real_pin-blind:+.4f})",
          flush=True)

    anchor_err = np.mean([np.linalg.norm(pred[ANCHOR_INDICES] - truth[ANCHOR_INDICES], axis=1).mean()
                          for _, _, truth, pred, _ in cache])
    print(f"\n(for reference: mean anchor error in these predictions was {anchor_err:.3f}mm)")
    print("This is the realistic number -- no oracle, no synthetic noise, actual")
    print("predictions from the confirmed 1.3736mm pipeline.")


if __name__ == "__main__":
    main()
