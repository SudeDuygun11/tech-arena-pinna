"""How much does the pipeline depend on the test mesh arriving in the SAME
global coordinate frame as the training meshes?

WHY THIS MATTERS
----------------
Two places bake in an absolute frame:

  1. pipeline.py crops the working region with `template.global_crop_center`,
     an absolute 3D point averaged over TRAINING subjects. If a test head sits
     somewhere else in space, that crop grabs the wrong surface -- or nothing.

  2. registration's rigid ICP starts from the identity transform, so it is a
     LOCAL optimiser. It refines a roughly-correct pose; it cannot recover a
     large rotation.

Both are fine if the organisers' test meshes use the same convention as ours
(they described aligning heads to the interaural axis, which suggests they do).
Neither is verified. This measures the tolerance so the assumption is a known
quantity rather than a hope.

WHAT IS MEASURED
----------------
Registration-only prediction (Stage 1a, no learned correction) for one held-out
ear, with the mesh AND its ground truth rotated together by increasing angles.
Rotating both keeps the comparison fair: a perfectly frame-independent pipeline
would score identically at every angle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.dataset import Dataset
from src.pipeline import predict_canonical
from src.registration.template import build_template


def rotation(axis, deg):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-template", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--k-references", type=int, default=3)
    ap.add_argument("--angles", type=float, nargs="+",
                    default=[0, 2, 5, 10, 20, 45, 90])
    ap.add_argument("--shift", type=float, nargs="+", default=[0, 2, 5, 20])
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    train_ids = ds.subject_ids[:args.n_template]
    test_ids = ds.subject_ids[args.n_template:args.n_template + args.n_test]

    print(f"template from {len(train_ids)} subjects, testing on {len(test_ids)}")
    template = build_template(ds, train_ids, k_references=args.k_references)

    axis = np.array([0.3, 1.0, 0.2])          # arbitrary, not an anatomical axis

    print(f"\n{'perturbation':>22s} {'mean 85-pt error':>18s}")
    print("-" * 42)
    for deg in args.angles:
        R = rotation(axis, deg)
        errs = []
        for ex in iter_landmarks_only(ds, test_ids):
            mesh = load_canonical_mesh(ds, ex.subject_id, ex.side)
            gt0 = np.asarray(ex.landmarks, float)
            # Rotate ABOUT THE EAR, not about the origin. The head sits far from
            # the origin, so rotating about it also translates the ear by tens of
            # mm -- which conflates two different perturbations and grossly
            # overstates the rotational sensitivity.
            pivot = gt0.mean(axis=0)
            mesh.vertices = (np.asarray(mesh.vertices) - pivot) @ R.T + pivot
            gt = (gt0 - pivot) @ R.T + pivot
            pred = predict_canonical(template, None, mesh)     # Stage 1a only
            errs.append(np.linalg.norm(pred - gt, axis=1).mean())
            del mesh
        print(f"{('rotate %5.1f deg' % deg):>22s} {np.mean(errs):15.3f} mm", flush=True)

    for mm in args.shift:
        if mm == 0:
            continue
        t = np.array([mm, 0.0, 0.0])
        errs = []
        for ex in iter_landmarks_only(ds, test_ids):
            mesh = load_canonical_mesh(ds, ex.subject_id, ex.side)
            mesh.vertices = np.asarray(mesh.vertices) + t
            gt = np.asarray(ex.landmarks, float) + t
            pred = predict_canonical(template, None, mesh)
            errs.append(np.linalg.norm(pred - gt, axis=1).mean())
            del mesh
        print(f"{('translate %4.1f mm' % mm):>22s} {np.mean(errs):15.3f} mm", flush=True)

    print("\nA frame-independent pipeline would give the SAME number on every row.")
    print("Growth means the test set must arrive in the training coordinate convention.")


if __name__ == "__main__":
    main()
