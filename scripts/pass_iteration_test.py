"""Does the anchor-pin / re-register loop COMPOUND?

The two-pass end-to-end confirmation landed at 1.3653mm against a 1.3736mm
one-pass baseline -- a -0.0083mm move with t = -0.87 over 400 paired ears,
i.e. a null. Before abandoning the mechanism there is one cheap question left:
a single extra pass may simply be too few. Each pass feeds better anchors into
the pose fit, which should feed better anchors back out, and that loop might
need three or four turns to converge somewhere useful -- or it might diverge.

This is now cheap because cross_validate.py was run with --save-models: the
fold-3 anchor and polish ensembles are already trained and sitting in
assets/bundle_twopass.pt, so sweeping the pass count needs inference only.

LEAKAGE CARE
------------
The bundle is fold 3 of a 3-fold subject-wise split with seed 0. This script
reconstructs that exact split and scores ONLY that fold's held-out subjects,
so no scored ear was seen during training. The fold index is read from the
bundle's own config rather than assumed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.estimator import _build
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS, N_ANCHORS, N_LANDMARKS
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split
from src.pipeline import predict_canonical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bundle", default="assets/bundle_twopass.pt")
    ap.add_argument("--passes", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n-ears", type=int, default=40,
                    help="held-out ears to score (0 = all of the fold)")
    args = ap.parse_args()

    b = torch.load(args.bundle, map_location="cpu", weights_only=False)
    cfg = b.get("config", {})
    template = b["template"]
    anchor_models = [_build(s, N_ANCHORS) for s in b["anchor_models"]]
    polish_models = [_build(s, N_LANDMARKS) for s in b.get("polish_models", [])]
    fold_i = cfg.get("outer_fold", 2)
    n_subjects = cfg.get("n_subjects", 200)
    print(f"bundle: {len(anchor_models)} anchor + {len(polish_models)} polish models, "
          f"outer_fold={fold_i}, n_subjects={n_subjects}", flush=True)

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    subject_ids = ds.subject_ids[:n_subjects]
    splits = list(k_fold_subject_split(subject_ids, 3, 0))
    _, test_ids = splits[fold_i]
    print(f"reconstructed fold {fold_i}: {len(test_ids)} held-out subjects", flush=True)

    cache = []
    for e in iter_landmarks_only(ds, test_ids):
        cache.append((np.asarray(e.landmarks, float),
                      load_canonical_mesh(ds, e.subject_id, e.side)))
        if args.n_ears and len(cache) >= args.n_ears:
            break
    print(f"  {len(cache)} ears cached\n", flush=True)

    anchor = np.zeros(N_LANDMARKS, bool)
    anchor[list(ANCHOR_INDICES)] = True

    print(f"{'passes':>7s} {'overall':>9s} {'anchors':>9s} {'non-anch':>9s} "
          f"{'inner_hx':>9s} {'lm74':>8s} {'secs':>7s}")
    base = None
    for n in args.passes:
        t0 = time.time()
        E = []
        for L, mesh in cache:
            pred = predict_canonical(template, anchor_models, mesh,
                                      polish_model=polish_models or None,
                                      registration_method=cfg.get("registration_method", "tps"),
                                      gaussian=cfg.get("gaussian_curvature", False),
                                      geodesic=cfg.get("geodesic_patch", False),
                                      patch_points=cfg.get("patch_points"),
                                      patch_radius=cfg.get("patch_radius"),
                                      n_passes=n)
            E.append(np.linalg.norm(pred - L, axis=1))
        E = np.array(E)                       # (n_ears, 85)
        ih = CONTOUR_SPECS["inner_helix"]["anchors"]
        ih_idx = np.arange(ih[0], ih[-1] + 1)
        o = E.mean()
        if base is None:
            base = o
        print(f"{n:7d} {o:9.4f} {E[:, anchor].mean():9.4f} {E[:, ~anchor].mean():9.4f} "
              f"{E[:, ih_idx].mean():9.4f} {E[:, 74].mean():8.4f} {time.time()-t0:7.0f}"
              f"   ({o-base:+.4f} vs 1 pass)", flush=True)

    print("\nIf the column is flat past 2, the loop does not compound and the")
    print("mechanism is finished. If it keeps falling, the end-to-end run simply")
    print("stopped one turn too early.")


if __name__ == "__main__":
    main()
