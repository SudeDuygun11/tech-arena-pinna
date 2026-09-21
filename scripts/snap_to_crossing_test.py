"""Does snapping a landmark-0 prediction onto the nearest curvature zero
crossing improve it, and from how far away?

BACKGROUND
----------
scripts/point0_sulcus_test.py established, over 40 ears, that ground-truth
landmark 0 sits 0.688 +- 0.469mm from the nearest zero crossing of smoothed
curvature, while its own neighbours 1-5 sit 2.978mm away and random nearby
surface points 2.013mm. So the crossing set carries real information, and
0.688mm is the ORACLE -- the best any "landmark 0 lies on a crossing" rule can
do, if it always picks the right crossing.

Walking the crest to find that crossing failed badly (6.7mm, a third of walks
found nothing). That left the cheaper question: given a prediction that is
already roughly right, is the NEAREST crossing the right one?

WHY PERTURBED GROUND TRUTH RATHER THAN REAL PREDICTIONS
-------------------------------------------------------
Real model predictions would need a full cross-validation run to generate. What
actually matters here is the capture radius -- how far off a starting guess can
be while the nearest crossing is still the correct one -- and that is measured
directly by displacing ground truth by a known amount.

The important caveat: this perturbation is ISOTROPIC, while real model error is
structured (it correlates along the contour, and the wrong-ridge failure is a
specific direction, not a random one). A wrong-ridge prediction sits near a
DIFFERENT ridge that has its own crossings, so real-world snapping will do worse
than these numbers. Read this as an upper bound on how well snapping can work.

Starting offsets are placed on the surface, since a prediction that has been
through the pipeline lands on or near the mesh rather than floating in space.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.dataset import Dataset
from scripts.point0_sulcus_test import concavity, smooth_field, zero_crossing_points

P0, P6 = 0, 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-ears", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=float, default=1.0,
                    help="curvature smoothing radius; 1.0 was best in the oracle test")
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--trials", type=int, default=20, help="random directions per offset")
    ap.add_argument("--crop", type=float, default=45.0)
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(ds.subject_ids), size=min(args.n_ears, len(ds)), replace=False)

    after = {o: [] for o in args.offsets}     # error after snapping
    before = {o: [] for o in args.offsets}    # error of the starting guess (on-surface)
    won = {o: 0 for o in args.offsets}
    tot = {o: 0 for o in args.offsets}
    oracle = []

    for n, si in enumerate(picks, 1):
        sid = ds.subject_ids[si]
        try:
            mesh, lm_left, _ = ds[si]
        except Exception as exc:
            print(f"[{n:3d}/{len(picks)}] {sid}: load failed ({exc})")
            continue
        L = np.asarray(lm_left, dtype=float)

        v_all = np.asarray(mesh.vertices)
        keep = np.linalg.norm(v_all - L[P6], axis=1) <= args.crop
        fmask = keep[np.asarray(mesh.faces)].all(axis=1)
        sub = mesh.submesh([np.flatnonzero(fmask)], append=True)
        verts = np.asarray(sub.vertices, dtype=float)
        faces = np.asarray(sub.faces)
        norms = np.asarray(sub.vertex_normals, dtype=float)
        del mesh, sub, v_all

        tree = cKDTree(verts)
        cur = smooth_field(tree, verts, concavity(verts, faces, norms), args.radius)
        cur = cur / (np.abs(cur).mean() + 1e-12)

        zc = zero_crossing_points(verts, faces, cur)
        if len(zc) == 0:
            continue
        zt = cKDTree(zc)
        oracle.append(float(zt.query(L[P0])[0]))

        for o in args.offsets:
            for _ in range(args.trials):
                d = rng.normal(size=3)
                d /= np.linalg.norm(d)
                guess = verts[tree.query(L[P0] + o * d)[1]]     # land it on the surface
                e_before = float(np.linalg.norm(guess - L[P0]))
                snapped = zc[zt.query(guess)[1]]
                e_after = float(np.linalg.norm(snapped - L[P0]))
                before[o].append(e_before)
                after[o].append(e_after)
                tot[o] += 1
                won[o] += (e_after < e_before)

        print(f"[{n:3d}/{len(picks)}] {sid}  oracle={oracle[-1]:.3f}mm  "
              f"crossings={len(zc)}", flush=True)

    print("\n" + "=" * 78)
    print(f"n ears = {len(oracle)}   smoothing radius = {args.radius}mm")
    print(f"ORACLE (nearest crossing to ground truth): {np.mean(oracle):.3f} "
          f"+- {np.std(oracle):.3f} mm  -- no rule can beat this")
    print("\nSNAP: displace ground truth, put it on the surface, take the nearest crossing")
    print(f"{'offset':>7s} {'start err':>10s} {'after snap':>12s} {'median':>9s} "
          f"{'p90':>8s} {'improved':>10s}")
    for o in args.offsets:
        b = np.array(before[o]); a = np.array(after[o])
        print(f"{o:7.1f} {b.mean():10.3f} {a.mean():12.3f} {np.median(a):9.3f} "
              f"{np.percentile(a,90):8.3f} {100*won[o]/max(tot[o],1):9.1f}%")
    print("\nsnapping is worth doing only where 'after snap' beats 'start err';")
    print("the offset at which that stops is the capture radius.")


if __name__ == "__main__":
    main()
