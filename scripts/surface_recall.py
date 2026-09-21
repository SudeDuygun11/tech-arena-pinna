"""Evaluate a preprocessing FILTER by surface recall.

Adopted from a teammate's diagnostic. The question a filter must answer is
"did I throw the answer away?", and the right way to ask it is: what fraction
of ground-truth landmarks lie within X mm of a RETAINED TRIANGLE?

Two things this deliberately does and does not tell you:
  - It DOES tell you whether the surface containing the answer survived.
  - It does NOT tell you whether the answer can be FOUND in what survived.
    A filter keeping 28% of faces at 99.6% recall still leaves thousands of
    candidate vertices; recall says the landmark is in there, not which one it
    is. Do not read high recall as low landmark error -- our own pipeline
    error (~1.58mm) is dominated by localisation, not by filtering losses.

Filters are scored on distance-to-triangle (see geometry.nearest_surface_points),
NOT distance-to-vertex, which overstates distance and is not comparable.

Usage:
    python scripts/surface_recall.py --data-dir "<parent of mesh/ landmarks/>" \
        --n-subjects 20 --filters crest80,crest90,concave25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.dataset import Dataset
from src.foundations.geometry import nearest_surface_points
from src.correction.patch_features import _local_curvature, crop_submesh
from src.experimental.ridge_filter import ridge_face_mask

TOLERANCES = [0.25, 0.50, 1.00]


def face_mask_from_vertex_mask(faces: np.ndarray, vmask: np.ndarray, min_votes: int = 2):
    return vmask[faces].sum(axis=1) >= min_votes


def build_filter(name: str, local, curv):
    """Return a boolean face mask for the named filter."""
    absc = np.abs(curv)
    if name.startswith("crest"):          # crest80 = top 20% |curvature|
        pct = float(name[5:])
        return face_mask_from_vertex_mask(local.faces, absc >= np.percentile(absc, pct))
    if name.startswith("concave"):        # concave25 = most-concave 25%
        pct = float(name[7:])
        return face_mask_from_vertex_mask(local.faces, curv <= np.percentile(curv, pct))
    if name.startswith("convex"):
        pct = float(name[6:])
        return face_mask_from_vertex_mask(local.faces, curv >= np.percentile(curv, 100 - pct))
    if name.startswith("corridor"):
        # corridor[:PCT[:MINCOMP[:RINGS]]]  e.g. corridor75:60:3
        parts = name.split(":")
        kw = {}
        if len(parts) > 1 and parts[1]:
            kw["percentile"] = float(parts[1])
        if len(parts) > 2 and parts[2]:
            kw["min_component"] = int(parts[2])
        if len(parts) > 3 and parts[3]:
            kw["dilate_rings"] = int(parts[3])
        return ridge_face_mask(local, curv=curv, **kw)
    if name == "all":
        return np.ones(len(local.faces), dtype=bool)
    raise ValueError(f"unknown filter: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=20)
    p.add_argument("--crop-radius", type=float, default=45.0)
    p.add_argument("--filters", default="all,crest80,crest90,crest95,concave25",
                    help="comma-separated: all | crestNN | concaveNN | convexNN")
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    names = [f.strip() for f in args.filters.split(",") if f.strip()]

    kept = {n: [] for n in names}
    dists = {n: [] for n in names}
    n_ears = 0

    for ex in iter_landmarks_only(ds, ds.subject_ids[:args.n_subjects]):
        mesh = load_canonical_mesh(ds, ex.subject_id, ex.side)
        local = crop_submesh(mesh, ex.landmarks.mean(axis=0), args.crop_radius)
        if len(local.vertices) < 500:
            continue
        n_ears += 1
        curv = _local_curvature(local, np.arange(len(local.vertices)))
        V, F = np.asarray(local.vertices), np.asarray(local.faces)

        for name in names:
            fmask = build_filter(name, local, curv)
            kept[name].append(fmask.mean())
            if not fmask.any():
                dists[name].extend([np.inf] * len(ex.landmarks))
                continue
            _, d = nearest_surface_points(V, F[fmask], ex.landmarks)
            dists[name].extend(d.tolist())

    print(f"\near{'s' if n_ears != 1 else ''} analysed: {n_ears}   "
          f"landmarks per filter: {len(dists[names[0]])}\n")
    hdr = (f"{'filter':14s}{'faces kept':>12s}" +
           "".join(f"recall<={t:.2f}mm".rjust(16) for t in TOLERANCES) +
           f"{'p95':>10s}{'worst':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        d = np.array(dists[name])
        row = f"{name:14s}{np.mean(kept[name]) * 100:11.1f}%"
        for t in TOLERANCES:
            row += f"{(d <= t).mean() * 100:15.2f}%"
        row += f"{np.percentile(d, 95):9.3f}mm{d.max():9.3f}mm"
        print(row)

    print("\nrecall<=X = share of GT landmarks within X mm of a retained triangle.")
    print("'worst' is the single largest distance -- a large value means the filter")
    print("EXCLUDED a true landmark on some ear, which no later stage can undo.")


if __name__ == "__main__":
    main()
