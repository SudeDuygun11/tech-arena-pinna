"""Step 2: does the shape prior interpolate the 70 non-anchor points better
than blend_correct does?

82% of the scored points are not predicted -- they come from interpolation. Two
independent findings say that is where the remaining error lives:

  * mesh-free test: given GROUND-TRUTH anchors, the PCA prior reconstructs the
    unobserved points to 0.735mm (all 85: 0.640mm). The project record puts
    blend_correct with GT anchors at ~0.9-1.6mm per contour.
  * the 400-ear prior run: anchors improved -0.025mm yet overall moved only
    -0.007mm, because the gain failed to propagate through blending.

Both point the same way, but they were never compared like-for-like on the same
ears with the same inputs. This does that: feed IDENTICAL ground-truth anchors
into each interpolation scheme and score only the 70 points neither scheme was
told about.

Using GT anchors is deliberate -- it isolates interpolation quality from anchor
error. The numbers are ceilings, not achievable pipeline errors.

Also reports a HYBRID: prior reconstruction followed by the surface snap, since
the prior has no notion of the mesh surface while blend_correct does, and
landmarks provably lie on it (0.0132mm).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.spatial import cKDTree

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS, N_LANDMARKS
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid, kabsch
from src.interpolation.interpolation import DEFAULT_SNAP_RADIUS, blend_correct
from src.registration.registration import compute_raw_full
from src.foundations.splits import k_fold_subject_split
from src.registration.template import build_template

N_MODES = 45
RIDGE = 0.05
NON_ANCHOR = np.setdiff1d(np.arange(N_LANDMARKS), ANCHOR_INDICES)


def procrustes_align(shapes, n_iter=5):
    aligned = shapes.copy()
    mean = aligned[0]
    for _ in range(n_iter):
        for i in range(len(aligned)):
            R, t = kabsch(aligned[i], mean)
            aligned[i] = apply_rigid(aligned[i], R, t)
        mean = aligned.mean(axis=0)
    return aligned, mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=60)
    p.add_argument("--k-references", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    ids = ds.subject_ids[:args.n_subjects]
    train_ids, test_ids = next(iter(k_fold_subject_split(ids, 2, args.seed)))
    test_set = set(test_ids)
    print(f"template+prior from {len(train_ids)} subjects; scoring {len(test_ids)}")

    template = build_template(ds, train_ids, k_references=args.k_references)
    shapes = np.array([e.landmarks for e in iter_landmarks_only(ds, train_ids)])
    aligned, mean_shape = procrustes_align(shapes.copy())
    X = aligned.reshape(len(aligned), -1)
    mu = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:N_MODES]
    evals = (S[:N_MODES] ** 2) / (len(X) - 1)
    rows = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in ANCHOR_INDICES])
    Po = P[:, rows]
    A = Po @ Po.T + RIDGE * np.diag(1.0 / np.maximum(evals, 1e-9))
    print(f"prior: {len(shapes)} shapes, {N_MODES} modes\n")

    res = {k: [] for k in ("blend", "prior", "prior_snap")}
    per_contour = {k: {c: [] for c in CONTOUR_SPECS} for k in res}

    for ex in iter_landmarks_only(ds, test_ids):
        mesh = load_canonical_mesh(ds, ex.subject_id, ex.side)
        raw = compute_raw_full(template, mesh.vertices, method="tps")
        gt = ex.landmarks
        gt_anchors = {a: gt[a] for a in ANCHOR_INDICES}

        # --- arm 1: current blending, given perfect anchors
        blended = blend_correct(raw, gt_anchors, mesh.vertices)

        # --- arm 2: PCA prior fitted to the same perfect anchors
        obs = np.array([gt[a] for a in ANCHOR_INDICES])
        R, t = kabsch(obs, mean_shape[ANCHOR_INDICES])
        obs_a = apply_rigid(obs, R, t)
        b = np.linalg.solve(A, Po @ (obs_a.reshape(-1) - mu[rows]))
        rec_a = (mu + P.T @ b).reshape(-1, 3)
        rec = apply_rigid(rec_a, R.T, -t @ R)     # back to mesh frame

        # --- arm 3: prior + the same surface snap blending uses
        tree = cKDTree(mesh.vertices)
        d, idx = tree.query(rec)
        snapped = rec.copy()
        close = d < DEFAULT_SNAP_RADIUS
        snapped[close] = mesh.vertices[idx][close]

        for name, pts in (("blend", blended), ("prior", rec), ("prior_snap", snapped)):
            e = np.linalg.norm(pts - gt, axis=1)
            res[name].append(e[NON_ANCHOR].mean())
            for c, spec in CONTOUR_SPECS.items():
                s, t2 = spec["range"]
                sel = [i for i in range(s, t2) if i in set(NON_ANCHOR.tolist())]
                per_contour[name][c].append(e[sel].mean())

    n = len(res["blend"])
    print(f"ears: {n}   scoring the {len(NON_ANCHOR)} NON-ANCHOR points only\n")
    print(f"{'method':>14s}{'mean':>10s}{'median':>10s}{'p95':>10s}")
    print("-" * 44)
    for k in ("blend", "prior", "prior_snap"):
        a = np.array(res[k])
        print(f"{k:>14s}{a.mean():9.4f}m{np.median(a):9.4f}m{np.percentile(a, 95):9.4f}m")

    # paired: identical anchors and ears in every arm, so this is a clean comparison
    bl = np.array(res["blend"])
    for k in ("prior", "prior_snap"):
        d = np.array(res[k]) - bl
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"\n{k} vs blend: {d.mean():+.4f}mm +/- {se:.4f} (SE)   "
              f"95% CI [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]")
        print(f"  ears improved {(d < 0).sum()}/{len(d)}   "
              f"{'SIGNIFICANT' if abs(d.mean()) > 1.96 * se else 'not significant'}")

    print(f"\n{'contour':>22s}" + "".join(k.rjust(13) for k in ("blend", "prior", "prior_snap")))
    print("-" * 61)
    for c in CONTOUR_SPECS:
        print(f"{c:>22s}" + "".join(f"{np.mean(per_contour[k][c]):12.4f}m"
                                     for k in ("blend", "prior", "prior_snap")))

    print("\nThese are CEILINGS (perfect anchors). Real pipeline error is higher.")
    print("What matters is which interpolation scheme wins, and by how much.")


if __name__ == "__main__":
    main()
