"""Do pretrained DINOv2 image features locate pinna landmarks?

Motivation: 2025 anatomical-landmark work uses frozen DINOv2 features (with
graph refinement), and DINO features are known for zero-shot semantic
correspondence. Rounds 1-3 showed local 3D surface descriptors cannot locate
the sliding anchors (74, 6, 22) and relational correction saturates near
-0.045mm. A 2D foundation model sees the ear as an image -- the way the
organizer's annotators saw shaded renders -- and was trained on far more data
than we have, so it is a genuinely different source of information.

PROTOCOL (no ground truth at test time)
  * render each ear head-on (orthographic, shaded by surface normal), FRAMED ON
    OUR PREDICTED LANDMARKS, 448px -> DINOv2 ViT-S/14 patch features (32x32x384)
  * per anchor, a feature PROTOTYPE = mean normalised feature at the TRUE
    landmark's pixel over the TRAINING-fold ears
  * test ear: cosine-similarity map, bilinear-upsampled, searched within 6mm of
    OUR PREDICTED anchor; argmax and soft-argmax -> depth buffer -> 3D
  * scored alone, as a per-anchor blend (step chosen on training folds), and as
    an extra input block to the V1 relational correction
  * same 3 outer subject folds that produced the saved predictions

LIMITS
  * one head-on view only (a multi-view version would recover more occluded
    points); ViT-S/14 is the smallest DINOv2
  * patch size 14px ~ 2.6mm at this framing, recovered sub-patch by upsampling
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from feature_probe_r2 import fit_predict
from feature_probe_r3 import load_sorted, rebuild, AI, KIND, STEPS
from src.foundations.canonical import load_canonical_mesh
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split
from src.multiview.multiview import densify, ear_frame, fill_holes, project, splat_render

RES = 448
MARGIN = 1.25
SEARCH_MM = 6.0
DATA = ROOT / "2026 Munich Tech Arena - Datas"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def backproject_many(view, px, py, depth):
    r, u, f = view["basis"]; res, ext = view["res"], view["extent"]
    x = ((np.asarray(px) / (res - 1)) * 2.0 - 1.0) * ext
    y = ((0.5 - np.asarray(py) / (res - 1)) * 2.0) * ext
    return view["target"] + x[:, None] * r + y[:, None] * u + np.asarray(depth)[:, None] * f


def sample_feat(fmap, px, py):
    """Bilinear sample a (C, 32, 32) patch-feature map at pixel coords."""
    g = fmap.shape[-1]
    gx = (np.asarray(px) / 14.0 - 0.5) / (g - 1) * 2 - 1
    gy = (np.asarray(py) / 14.0 - 0.5) / (g - 1) * 2 - 1
    grid = torch.tensor(np.stack([gx, gy], -1), dtype=torch.float32).view(1, 1, -1, 2)
    s = TF.grid_sample(fmap[None].float(), grid, align_corners=True)[0, :, 0].T   # (m, C)
    return TF.normalize(s, dim=1)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    if args.limit:                                   # smoke test only
        pred, truth, subj, side, keys = (a[:args.limit] for a in (pred, truth, subj, side, keys))
    n = len(pred)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    folds = list(k_fold_subject_split(ds.subject_ids[:200], 3, 0))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(folds):
        fold_of[np.isin(subj, list(te))] = f

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).eval().to(dev)

    feats, views, ok = [], [], np.ones(n, bool)
    t0 = time.time()
    for i in range(n):
        try:
            mesh = load_canonical_mesh(ds, subj[i], side[i])
        except Exception as exc:
            print(f"  skip {keys[i]}: {exc}"); ok[i] = False; feats.append(None); views.append(None); continue
        c0 = pred[i].mean(0)
        va = np.asarray(mesh.vertices)
        keep = np.linalg.norm(va - c0, axis=1) <= 50.0
        sub = mesh.submesh([np.flatnonzero(keep[np.asarray(mesh.faces)].all(1))], append=True)
        del mesh, va; gc.collect()
        V = np.asarray(sub.vertices, float); F = np.asarray(sub.faces); N = np.asarray(sub.vertex_normals, float)
        centre, basis = ear_frame(pred[i], V)                      # PREDICTED landmarks frame the view
        extent = np.abs((pred[i] - centre) @ basis.T).max() * MARGIN
        P, NN, C = densify(V, F, N, np.zeros(len(V)))
        eye = centre + basis[2] * extent * 6.0
        vw = fill_holes(splat_render(P, NN, C, eye, centre, basis[0], extent, RES))
        shade = np.clip((vw["normal"] * basis[2]).sum(-1), 0, 1) * vw["mask"]
        img = torch.tensor(shade, dtype=torch.float32)[None, None].repeat(1, 3, 1, 1)
        img = ((img - MEAN) / STD).to(dev)
        with torch.no_grad():
            tok = model.forward_features(img)["x_norm_patchtokens"][0]
        g = int(np.sqrt(tok.shape[0]))
        feats.append(tok.T.reshape(-1, g, g).half().cpu())
        views.append({"basis": vw["basis"], "target": vw["target"], "extent": vw["extent"],
                      "res": RES, "depth": vw["depth"].astype(np.float32), "mask": vw["mask"]})
        del sub, V, F, N, P, NN, C, vw; gc.collect()
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n} ears rendered+encoded  {el/60:.1f} min  (~{el/(i+1)*(n-i-1)/60:.0f} min left)", flush=True)

    # pixel coordinates of true and predicted anchors in each ear's view
    tpx = np.full((n, len(AI), 2), np.nan); ppx = np.full((n, len(AI), 2), np.nan)
    for i in range(n):
        if ok[i]:
            tpx[i] = project(views[i], truth[i, AI]); ppx[i] = project(views[i], pred[i, AI])

    dino_arg = pred[:, AI].copy(); dino_soft = pred[:, AI].copy()
    for f in range(3):
        tr = np.flatnonzero((fold_of != f) & ok); te = np.flatnonzero((fold_of == f) & ok)
        protos = torch.stack([torch.stack([sample_feat(feats[i], tpx[i, k:k+1, 0], tpx[i, k:k+1, 1])[0]
                                           for i in tr]).mean(0) for k in range(len(AI))])
        protos = TF.normalize(protos, dim=1)                                        # (15, 384)
        for i in te:
            vw = views[i]
            fm = TF.normalize(feats[i].float(), dim=0)                         # (384, 32, 32)
            sims32 = torch.einsum("chw,kc->khw", fm, protos)                    # (15, 32, 32)
            # upsample the 15 similarity maps, not the 384-channel features (~300MB/ear)
            sims = TF.interpolate(sims32[None], size=(RES, RES), mode="bilinear", align_corners=False)[0].numpy()
            mm_per_px = 2.0 * vw["extent"] / (RES - 1)
            rad = SEARCH_MM / mm_per_px
            yy, xx = np.mgrid[0:RES, 0:RES]
            for k in range(len(AI)):
                cx, cy = ppx[i, k]
                win = ((xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2) & vw["mask"] & np.isfinite(vw["depth"])
                if win.sum() < 5:
                    continue
                sim = sims[k]
                ys, xs = np.nonzero(win); s = sim[ys, xs]
                j = int(np.argmax(s))
                dino_arg[i, k] = backproject_many(vw, [xs[j]], [ys[j]], [vw["depth"][ys[j], xs[j]]])[0]
                wts = np.exp((s - s.max()) / 0.02); wts /= wts.sum()
                sx, sy = (wts * xs).sum(), (wts * ys).sum()
                ix, iy = int(round(sx)), int(round(sy))
                dpt = vw["depth"][iy, ix] if np.isfinite(vw["depth"][iy, ix]) else vw["depth"][ys[j], xs[j]]
                dino_soft[i, k] = backproject_many(vw, [sx], [sy], [dpt])[0]
        print(f"  fold {f+1}: prototypes from {len(tr)} ears, matched {len(te)} ears", flush=True)

    PA, TA = pred[:, AI], truth[:, AI]
    base = np.linalg.norm(pred - truth, axis=2)
    print("\nper-anchor error (mm): our prediction vs DINOv2 match within 6mm")
    print(f"  {'anchor':>6s} {'kind':>11s} {'ours':>7s} {'argmax':>7s} {'soft':>7s}")
    for k, gi in enumerate(AI):
        print(f"  {gi:6d} {KIND[gi]:>11s} {np.linalg.norm(PA[:, k]-TA[:, k], axis=1).mean():7.3f} "
              f"{np.linalg.norm(dino_arg[:, k]-TA[:, k], axis=1).mean():7.3f} "
              f"{np.linalg.norm(dino_soft[:, k]-TA[:, k], axis=1).mean():7.3f}")

    r3 = np.load(ROOT / "results" / "feature_probe_r3.npz", allow_pickle=True)
    assert list(r3["keys"][:n]) == list(keys)
    v1_anch = pred.copy(); v1_anch[:, AI] = r3["anch_V1"][:n]
    v1_ear = np.linalg.norm(rebuild(pred, v1_anch) - truth, axis=2).mean(1)

    def score(name, full):
        step = full - PA; na = pred.copy()
        for f in range(3):
            tr, te = fold_of != f, fold_of == f
            for k in range(len(AI)):
                errs = [np.linalg.norm(PA[tr, k] + w * step[tr, k] - TA[tr, k], axis=1).mean() for w in STEPS]
                na[te, AI[k]] = PA[te, k] + STEPS[int(np.argmin(errs))] * step[te, k]
        e = np.linalg.norm(rebuild(pred, na) - truth, axis=2); ear = e.mean(1)
        d0 = ear - base.mean(1); d1 = ear - v1_ear
        print(f"  {name:36s} {e.mean():.4f}  vs base {d0.mean():+.4f} (t {d0.mean()/(d0.std(ddof=1)/np.sqrt(n)):+.2f})"
              f"  vs V1 {d1.mean():+.4f} (t {d1.mean()/(d1.std(ddof=1)/np.sqrt(n)):+.2f})"
              f"   74 {e[:, 74].mean():.3f}  6 {e[:, 6].mean():.3f}  22 {e[:, 22].mean():.3f}")
        return na

    print(f"\nend-to-end (baseline {base.mean():.4f}, V1 {v1_ear.mean():.4f}):")
    score("DINO argmax, per-anchor blend", dino_arg)
    score("DINO soft-argmax, per-anchor blend", dino_soft)
    prop = PA.copy()
    for k in range(len(AI)):
        X = np.hstack([(PA - PA[:, k][:, None, :]).reshape(n, -1),
                       (dino_soft - PA[:, k][:, None, :]).reshape(n, -1)])
        Y = TA[:, k] - PA[:, k]
        for f in range(3):
            tr, te = fold_of != f, fold_of == f
            prop[te, k] = PA[te, k] + fit_predict(X[tr], Y[tr], subj[tr], X[te])
    score("V1 relational + DINO block", prop)
    if args.limit:
        return
    np.savez_compressed(ROOT / "results" / "dino_probe.npz", keys=keys, dino_arg=dino_arg, dino_soft=dino_soft)
    print("\nsaved -> results/dino_probe.npz")


if __name__ == "__main__":
    main()
