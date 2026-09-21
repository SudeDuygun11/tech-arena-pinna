"""WHY is the error tangential? Can the network even SEE the difference?

68% of anchor squared error is along-contour sliding. The question this
answers is whether that is a training failure or an information failure.

The test: take the true landmark, then slide along the true contour by
-4..+4mm. At each slid position extract the same local patch the network
sees (radius 7mm, the pipeline's PATCH_RADIUS). Translate the slid patch
back by the slide vector and measure Chamfer distance to the patch at the
true position.

Interpretation. If the ear surface near a landmark is locally an EXTRUSION
along the contour -- a smooth ridge of roughly constant cross-section -- then
sliding and translating back lands on nearly the same point set, Chamfer stays
near zero, and the patch at +2mm is INDISTINGUISHABLE from the patch at 0.
No network of any capacity can resolve a difference that is not in its input.
That would make tangential error an information limit, and the fix is a bigger
or differently-shaped receptive field, not more training.

If instead Chamfer rises steeply with slide, the information IS present and
we are simply failing to use it -- a modelling failure, fixable by the 1-D
arc-offset head alone.

Compared across anchors of different definition types, since the error table
says junction-defined anchors (54: 26.9% along) behave very differently from
extremum- and construction-defined ones (74: 93.2%, 6: 89.1%).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correction.patch_features import PATCH_RADIUS, crop_submesh
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import CONTOUR_SPECS
from src.foundations.dataset import Dataset

# anchors chosen to span the %-along range measured on the saved predictions
PROBES = [(74, "constructed", 93.2), (6, "extremum", 89.1), (22, "extremum", 84.5),
          (64, "cross-ref", 66.0), (24, "junction", 40.8), (54, "junction", 26.9)]


def contour_point(L, gi, s):
    """Position s mm along the true contour from landmark gi (s may be negative)."""
    for spec in CONTOUR_SPECS.values():
        a, b = spec["range"]
        if a <= gi < b:
            break
    poly = L[a:b]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    here = cum[gi - a]
    want = np.clip(here + s, 0, cum[-1])
    return np.stack([np.interp(want, cum, poly[:, k]) for k in range(3)])


def chamfer(A, B):
    if len(A) < 10 or len(B) < 10:
        return np.nan
    da = cKDTree(B).query(A)[0]
    db = cKDTree(A).query(B)[0]
    return 0.5 * (da.mean() + db.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-ears", type=int, default=20)
    ap.add_argument("--slides", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--radius", type=float, default=PATCH_RADIUS)
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    print(f"patch radius {args.radius} mm (the pipeline's PATCH_RADIUS), "
          f"{args.n_ears} ears\n", flush=True)

    acc = {(gi, s): [] for gi, _, _ in PROBES for s in args.slides}
    done = 0
    for e in iter_landmarks_only(ds, ds.subject_ids):
        L = np.asarray(e.landmarks, float)
        mesh = load_canonical_mesh(ds, e.subject_id, e.side)
        for gi, _, _ in PROBES:
            sub = crop_submesh(mesh, L[gi], args.radius + 6.0)
            V = np.asarray(sub.vertices)
            if len(V) < 200:
                continue
            tree = cKDTree(V)
            p0 = L[gi]
            base = V[tree.query_ball_point(p0, args.radius)] - p0
            for s in args.slides:
                for sgn in (+1, -1):
                    p1 = contour_point(L, gi, sgn * s)
                    pts = V[tree.query_ball_point(p1, args.radius)] - p1
                    c = chamfer(base, pts)
                    if not np.isnan(c):
                        acc[(gi, s)].append(c)
        del mesh
        done += 1
        if done % 5 == 0:
            print(f"  {done} ears...", flush=True)
        if done >= args.n_ears:
            break

    print(f"\nChamfer distance (mm) between the patch AT the landmark and the patch")
    print(f"slid s mm along the contour, after translating back. Near 0 = the two")
    print(f"patches look the same, so the position is not identifiable from geometry.\n")
    hdr = "  ".join(f"{s:>5.1f}" for s in args.slides)
    print(f"  {'anchor':>6s} {'type':>12s} {'%along':>7s}   {hdr}")
    for gi, kind, pct in PROBES:
        row = "  ".join(f"{np.mean(acc[(gi, s)]):5.3f}" if acc[(gi, s)] else "  n/a"
                        for s in args.slides)
        print(f"  {gi:6d} {kind:>12s} {pct:6.1f}%   {row}")

    print("\nA row that stays low across all slides is an INFORMATION limit:")
    print("the network's input does not contain the answer. A row that climbs")
    print("steeply is a MODELLING limit: the signal is there and we are missing it.")


if __name__ == "__main__":
    main()
