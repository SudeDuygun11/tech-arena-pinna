"""Render every ear to a cached multi-view stack, once.

WHY CACHE
---------
Rendering itself is cheap (~58ms/view). The expensive part is loading a 15-34MB
head mesh and computing smoothed curvature over it: 7-14s per ear. Doing that
inside the training loop would cost roughly an hour per epoch, so the views are
computed once and stored.

WHAT IS STORED, per ear, per view
---------------------------------
    depth   (H,W) float16   distance along the view axis, relative to the frame
                            centre; NaN where nothing was hit
    normal  (H,W,3) float16 surface normal
    curv    (H,W) float16   signed smoothed curvature -- negative convex (ridge
                            crest), positive concave (sulcus)
    mask    (H,W) bool

The curvature channel is deliberate. Landmark 0's empirical signature lives in
it: measured over 40 ears, smoothed curvature at landmark 0 is 0.042 +- 0.665
(i.e. a sign change) against -2.085 at landmark 6, and landmark 0 sits 0.688mm
from the nearest zero crossing while random nearby surface sits 2.013mm away.
Handing the network that channel directly beats making it rediscover the cue
from depth alone.

Also stored per view: the camera basis, frame centre and extent (needed to
back-project a predicted pixel to 3D), and the projected pixel coordinates of
all 85 landmarks together with a per-landmark visibility flag. All 85 are stored
even though the first probe only uses landmark 0, so that extending to other
landmarks needs no re-render.

VISIBILITY
----------
A landmark counts as visible only if the rendered depth at its pixel matches the
landmark's own depth along the view axis (within `--vis-tol` mm). Testing only
that the pixel is covered is wrong and was measured to be wrong: it lets through
cases where the head occludes the ear, which inflated a round-trip error test
from 0.285mm to 1.322mm until the check was fixed.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.dataset import Dataset
from src.multiview.multiview import render_ear, project
from scripts.point0_sulcus_test import concavity, smooth_field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="parent of mesh/ and landmarks/")
    ap.add_argument("--out", default="cache/views", help="output directory")
    ap.add_argument("--n-subjects", type=int, default=200)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--n-lateral", type=int, default=8)
    ap.add_argument("--no-superior", action="store_true")
    ap.add_argument("--curv-radius", type=float, default=1.0,
                    help="curvature smoothing radius in mm; 1.0 gave the tightest "
                         "zero-crossing localisation (0.688mm) in the oracle test")
    ap.add_argument("--crop", type=float, default=60.0)
    ap.add_argument("--vis-tol", type=float, default=1.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    subject_ids = ds.subject_ids[:args.n_subjects]
    todo = [(e.subject_id, e.side, e.landmarks)
            for e in iter_landmarks_only(ds, subject_ids)]
    print(f"{len(todo)} ears from {len(subject_ids)} subjects -> {out}")

    t_start = time.time()
    written = skipped = 0
    for n, (sid, side, L) in enumerate(todo, 1):
        path = out / f"{sid}_{side}.npz"
        if path.exists() and not args.overwrite:
            skipped += 1
            continue

        L = np.asarray(L, dtype=float)
        try:
            mesh = load_canonical_mesh(ds, sid, side)
        except Exception as exc:
            print(f"[{n:4d}/{len(todo)}] {sid}_{side}: load failed ({exc})", flush=True)
            continue

        # crop to the ear, then free the full head immediately -- holding the
        # trimesh object with its cached arrays is what caused MemoryError
        # crashes in earlier full-scale runs
        v_all = np.asarray(mesh.vertices)
        keep = np.linalg.norm(v_all - L.mean(axis=0), axis=1) <= args.crop
        fmask = keep[np.asarray(mesh.faces)].all(axis=1)
        sub = mesh.submesh([np.flatnonzero(fmask)], append=True)
        verts = np.asarray(sub.vertices, dtype=float)
        faces = np.asarray(sub.faces)
        norms = np.asarray(sub.vertex_normals, dtype=float)
        del mesh, sub, v_all
        gc.collect()

        tree = cKDTree(verts)
        curv = smooth_field(tree, verts, concavity(verts, faces, norms), args.curv_radius)
        curv = curv / (np.abs(curv).mean() + 1e-12)          # per-ear scale normalise

        views, centre, basis, extent = render_ear(
            verts, faces, norms, curv, L,
            n_lateral=args.n_lateral, superior=not args.no_superior, res=args.res)

        depth = np.stack([v["depth"] for v in views]).astype(np.float16)
        normal = np.stack([v["normal"] for v in views]).astype(np.float16)
        cur_img = np.stack([v["curv"] for v in views]).astype(np.float16)
        mask = np.stack([v["mask"] for v in views])

        px = np.zeros((len(views), 85, 2), np.float32)
        vis = np.zeros((len(views), 85), bool)
        for vi, vw in enumerate(views):
            p = project(vw, L)
            px[vi] = p
            _, _, f = vw["basis"]
            own = (L - vw["target"]) @ f                      # each landmark's own depth
            c = np.rint(p[:, 0]).astype(int)
            r = np.rint(p[:, 1]).astype(int)
            inside = (c >= 0) & (c < args.res) & (r >= 0) & (r < args.res)
            cc, rr = np.clip(c, 0, args.res - 1), np.clip(r, 0, args.res - 1)
            d_at = vw["depth"][rr, cc]
            vis[vi] = inside & vw["mask"][rr, cc] & (np.abs(d_at - own) <= args.vis_tol)

        np.savez_compressed(
            path, depth=depth, normal=normal, curv=cur_img, mask=mask,
            px=px, vis=vis, landmarks=L.astype(np.float32),
            centre=centre.astype(np.float32), basis=basis.astype(np.float32),
            extent=np.float32(extent),
            dirs=np.stack([v["dir_local"] for v in views]).astype(np.float32),
            view_basis=np.stack([np.stack(v["basis"]) for v in views]).astype(np.float32),
            view_target=np.stack([v["target"] for v in views]).astype(np.float32))
        written += 1

        if n % 10 == 0 or n == len(todo):
            el = time.time() - t_start
            rate = el / max(written, 1)
            print(f"[{n:4d}/{len(todo)}] {sid}_{side}  "
                  f"{el/60:.1f}min elapsed, {rate:.1f}s/ear, "
                  f"~{rate*(len(todo)-n)/60:.0f}min left  "
                  f"lm0 visible in {vis[:, 0].sum()}/{len(views)} views", flush=True)

    total = sum(f.stat().st_size for f in out.glob("*.npz"))
    print(f"\nwrote {written}, skipped {skipped}  |  {total/1e9:.2f} GB in {out}")


if __name__ == "__main__":
    main()
