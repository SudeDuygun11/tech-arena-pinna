"""Are the ground-truth landmarks quantized to mesh vertices?

If the annotator placed landmarks by clicking mesh VERTICES, the labels are
quantized to the vertex spacing (~0.95mm here) and no model can be scored
below that resolution -- a sub-1mm target would then sit at or under the
precision of the targets themselves.

Test: distance from each GT landmark to its nearest mesh vertex, compared
against a NULL of points sampled uniformly on the same triangles. If the GT
distances are ~0 the labels are vertex-snapped. If they match the null, the
annotator placed points freely on the surface and vertex quantization is not
a limiting factor.

Uses left ears only: canonical mirroring is an exact sign flip, but left ears
need no transform at all, so this avoids the question entirely.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.spatial import cKDTree

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.dataset import Dataset
from src.foundations.geometry import nearest_surface_points
from src.correction.patch_features import crop_submesh

EXACT_TOL = 1e-6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=30)
    p.add_argument("--crop-radius", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    rng = np.random.default_rng(args.seed)

    d_vert, d_surf, d_null, edges = [], [], [], []
    n_ears = 0

    for ex in iter_landmarks_only(ds, ds.subject_ids[:args.n_subjects]):
        if ex.side != "left":
            continue
        mesh = load_canonical_mesh(ds, ex.subject_id, ex.side)
        local = crop_submesh(mesh, ex.landmarks.mean(axis=0), args.crop_radius)
        V = np.asarray(local.vertices)
        F = np.asarray(local.faces)
        if len(V) < 500:
            continue
        n_ears += 1

        edges.append(float(np.linalg.norm(
            V[local.edges_unique[:, 0]] - V[local.edges_unique[:, 1]], axis=1).mean()))

        tree = cKDTree(V)
        d_vert.extend(tree.query(ex.landmarks)[0].tolist())
        d_surf.extend(nearest_surface_points(V, F, ex.landmarks)[1].tolist())

        # NULL: points placed uniformly at random ON the triangles
        fi = rng.integers(len(F), size=len(ex.landmarks))
        tri = V[F[fi]]
        r1, r2 = rng.random(len(fi)), rng.random(len(fi))
        sq = np.sqrt(r1)
        bary = np.stack([1 - sq, sq * (1 - r2), sq * r2], axis=1)
        pts = (tri * bary[:, :, None]).sum(axis=1)
        d_null.extend(tree.query(pts)[0].tolist())

    dv, dn, dsuf = np.array(d_vert), np.array(d_null), np.array(d_surf)
    print(f"left ears analysed: {n_ears}   landmarks: {len(dv)}")
    print(f"mean mesh edge length: {np.mean(edges):.3f}mm\n")

    print("=== distance from GT landmark to nearest VERTEX ===")
    print(f"  mean {dv.mean():.4f}mm   median {np.median(dv):.4f}mm   "
          f"p95 {np.percentile(dv, 95):.4f}mm   max {dv.max():.4f}mm")
    print(f"  exactly on a vertex (<{EXACT_TOL}mm): {(dv < EXACT_TOL).mean() * 100:.2f}%")

    print("\n=== NULL: random points placed freely on the same triangles ===")
    print(f"  mean {dn.mean():.4f}mm   median {np.median(dn):.4f}mm   "
          f"p95 {np.percentile(dn, 95):.4f}mm")

    print("\n=== distance from GT landmark to nearest TRIANGLE SURFACE ===")
    print(f"  mean {dsuf.mean():.4f}mm   median {np.median(dsuf):.4f}mm   "
          f"p95 {np.percentile(dsuf, 95):.4f}mm")

    ratio = dv.mean() / dn.mean() if dn.mean() > 0 else np.inf
    print(f"\nGT-to-vertex / NULL-to-vertex ratio: {ratio:.3f}")
    print("  ~0.00  => labels are vertex-snapped; vertex spacing is a hard label floor")
    print("  ~1.00  => labels placed freely on the surface; no vertex quantization")


if __name__ == "__main__":
    main()
