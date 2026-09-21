"""Ceiling test for a 2D image model: what does the render -> back-project
round trip cost, given a PERFECT 2D curve?

The construction rules (6 = upper end of the outer helix's max-diameter
chord, 64 = its projection onto the inner helix, 74 = 10 steps along) are
exact on a true curve -- median 0.037 / 0.417 / 0.120 mm -- and useless on
ours, because ours is built from the very points the rules should produce.
An image model breaks that circularity: a curve SEGMENTATION has no
tangential degree of freedom, which is precisely the 68%-of-error direction
that point regression cannot resolve (a <2mm slide is below the mesh's own
0.430mm sampling in a 7mm patch).

But a 2D curve has to be lifted back to 3D through the depth buffer, and the
helix rim is exactly where the surface is most nearly tangent to the view
ray. This measures that loss ALONE, with no network: take the ground-truth
landmarks, project them into each view, read the depth the renderer stored,
back-project, and compare with where they started.

If a perfect 2D curve returns as a 1.5mm 3D curve, the approach is capped
above our current 1.3653mm and should be dropped. If it returns at ~0.2mm,
the ceiling is real and what remains is a training problem.

Reported separately:
  * per-view error, and error under ORACLE view selection (best view per
    landmark) -- what a multi-view fusion could reach
  * integer-pixel vs sub-pixel, since a real model predicts a pixel and the
    orthographic frame fixes mm-per-pixel
  * visibility, since a landmark seen in no view cannot be recovered at all
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES
from src.foundations.dataset import Dataset
from src.multiview.multiview import project, render_ear


def backproject_many(view, px, py, depth):
    """Vectorised inverse of multiview.project.

    multiview.backproject does the same arithmetic but writes `x * r`, which
    only broadcasts when x is scalar; here px/py/depth are arrays of landmarks,
    so the axes have to be made explicit."""
    r, u, f = view["basis"]
    res, ext = view["res"], view["extent"]
    x = ((np.asarray(px) / (res - 1)) * 2.0 - 1.0) * ext
    y = ((0.5 - np.asarray(py) / (res - 1)) * 2.0) * ext
    return (view["target"] + x[:, None] * r + y[:, None] * u
            + np.asarray(depth)[:, None] * f)


def bilinear(img, px, py):
    """Sample a float image at sub-pixel coordinates; NaN where any corner is
    not finite (the z-buffer stores inf for empty pixels)."""
    res = img.shape[0]
    x0 = np.clip(np.floor(px).astype(int), 0, res - 2)
    y0 = np.clip(np.floor(py).astype(int), 0, res - 2)
    fx, fy = px - x0, py - y0
    q = np.stack([img[y0, x0], img[y0, x0 + 1], img[y0 + 1, x0], img[y0 + 1, x0 + 1]])
    bad = ~np.isfinite(q).all(axis=0)
    q = np.where(np.isfinite(q), q, 0.0)
    out = (q[0]*(1-fx)*(1-fy) + q[1]*fx*(1-fy) + q[2]*(1-fx)*fy + q[3]*fx*fy)
    return np.where(bad, np.nan, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-ears", type=int, default=12)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--n-lateral", type=int, default=8)
    ap.add_argument("--crop", type=float, default=45.0)
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))

    per_view, oracle_sub, oracle_int, nvis, mmpx = [], [], [], [], []
    anchor_or = []
    done = 0
    for e in iter_landmarks_only(ds, ds.subject_ids):
        L = np.asarray(e.landmarks, float)
        try:
            mesh = load_canonical_mesh(ds, e.subject_id, e.side)
        except Exception:
            continue
        v_all = np.asarray(mesh.vertices)
        keep = np.linalg.norm(v_all - L.mean(0), axis=1) <= args.crop
        fmask = keep[np.asarray(mesh.faces)].all(axis=1)
        sub = mesh.submesh([np.flatnonzero(fmask)], append=True)
        verts = np.asarray(sub.vertices, float)
        faces = np.asarray(sub.faces)
        norms = np.asarray(sub.vertex_normals, float)
        del mesh, sub, v_all
        gc.collect()

        curv = np.zeros(len(verts))            # curvature unused by this test
        views, centre, basis, extent = render_ear(
            verts, faces, norms, curv, L, n_lateral=args.n_lateral, res=args.res)
        mmpx.append(2.0 * extent / (args.res - 1))

        errs_sub = np.full((len(views), 85), np.nan)
        errs_int = np.full((len(views), 85), np.nan)
        for vi, vw in enumerate(views):
            p = project(vw, L)
            px, py = p[:, 0], p[:, 1]
            inb = (px >= 0) & (px <= args.res - 1) & (py >= 0) & (py <= args.res - 1)
            d_sub = np.full(85, np.nan)
            d_sub[inb] = bilinear(vw["depth"], px[inb], py[inb])
            ok = np.isfinite(d_sub)
            if ok.any():
                rec = backproject_many(vw, px[ok], py[ok], d_sub[ok])
                errs_sub[vi, ok] = np.linalg.norm(rec - L[ok], axis=1)
            ix, iy = np.rint(px), np.rint(py)
            inb2 = (ix >= 0) & (ix <= args.res - 1) & (iy >= 0) & (iy <= args.res - 1)
            d_i = np.full(85, np.nan)
            idx = np.flatnonzero(inb2)
            d_i[idx] = vw["depth"][iy[idx].astype(int), ix[idx].astype(int)]
            ok2 = np.isfinite(d_i)
            if ok2.any():
                rec = backproject_many(vw, ix[ok2], iy[ok2], d_i[ok2])
                errs_int[vi, ok2] = np.linalg.norm(rec - L[ok2], axis=1)

        per_view.append(np.nanmean(errs_sub))
        nvis.append(np.isfinite(errs_sub).sum(0))
        with np.errstate(all="ignore"):
            best_sub = np.nanmin(errs_sub, axis=0)
            best_int = np.nanmin(errs_int, axis=0)
        oracle_sub.append(best_sub)
        oracle_int.append(best_int)
        anchor_or.append(best_sub[list(ANCHOR_INDICES)])

        done += 1
        print(f"  {done}/{args.n_ears} ears", flush=True)
        if done >= args.n_ears:
            break

    O = np.array(oracle_sub); Oi = np.array(oracle_int); V = np.array(nvis)
    print(f"\nrendered {done} ears, {args.n_lateral}+1 views, {args.res}px")
    print(f"scale: {np.mean(mmpx):.4f} mm per pixel  "
          f"(so +-0.5px quantisation = +-{np.mean(mmpx)/2:.4f} mm)\n")

    print(f"visibility: landmarks seen in >=1 view: "
          f"{100*(V > 0).mean():.2f}%   median views per landmark {np.median(V):.0f}")
    print(f"            landmarks seen in NO view: {(V == 0).sum()} of {V.size}\n")

    print("round-trip 3D error of a PERFECT 2D curve:")
    print(f"  mean over all views            {np.nanmean(per_view):.4f} mm")
    print(f"  ORACLE best view, sub-pixel    {np.nanmean(O):.4f} mm   "
          f"(median {np.nanmedian(O):.4f}, p90 {np.nanpercentile(O,90):.4f})")
    print(f"  ORACLE best view, integer px   {np.nanmean(Oi):.4f} mm")
    print(f"  ORACLE best view, 15 anchors   {np.nanmean(np.array(anchor_or)):.4f} mm")
    print(f"\n  our current pipeline: 1.3653 mm overall, 1.4698 on anchors")
    print("  A ceiling well under 1.0 means the image route has real headroom.")


if __name__ == "__main__":
    main()
