"""Multi-view probe: predict landmark 0 from rendered views, fuse to 3D.

THE QUESTION
------------
Three times now, a geometric cue for landmark 0 has been shown to contain the
answer while no selector could extract it:

    candidate ranking   oracle 0.534mm   actual selector 5.524mm
    curvature crossing  oracle 0.688mm   crest walk      6.717mm
    nearest-crossing snap                no better than the starting guess

The information is there; picking the right instance of it is what fails. The
organisers described the annotator's own guard against exactly this: "a point
may appear correctly positioned from one viewing direction while actually lying
towards the inner or outer side of the ridge when inspected from another." A
wrong ridge agrees with the truth in one view and disagrees in another, which is
a selection signal no single local patch can provide.

So: render the ear from 9 viewpoints, predict a 2D heatmap per view with an
ImageNet-pretrained backbone, and fuse the per-view answers into one 3D point.

COMPARATORS (all measured, same landmark)
-----------------------------------------
    0.285mm   rendering floor at 256px (project -> back-project, perfect prediction)
    0.688mm   zero-crossing oracle: nearest crossing to ground truth
    2.09mm    current pipeline, outer_helix contour average
    6.717mm   crest walk
   21.26mm    predict nothing, return landmark 6

SPLITS
------
Subject-wise, never ear-wise. Left and right ears of one subject are the same
person's anatomy mirrored, so splitting by ear would leak.

FUSION
------
Each view whose prediction lands on rendered surface gives a full 3D point (a
pixel plus its depth). Those are combined by a confidence-weighted mean after
rejecting outliers against the weighted median. Ray intersection was considered
and rejected: the 8 lateral views sit within 25 degrees of the ear normal, so
their rays are near-parallel and the intersection is ill-conditioned along
exactly the axis we care about. Measured depth has no such problem.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.multiview.resnet_heatmap import HeatmapNet, gaussian_target, soft_argmax

LM = 0                      # the landmark under test


class EarCache:
    """Lazy npz loader with a bounded in-memory cache.

    Preloading all 400 ears as float16 would be ~2.4GB resident; loading from
    disk every time costs ~30ms per ear. A capped dict gets most of the speed
    for a fraction of the memory.
    """

    def __init__(self, files, cap: int = 150):
        self.files = files
        self.cap = cap
        self._d = {}

    def __len__(self):
        return len(self.files)

    def get(self, i):
        if i in self._d:
            return self._d[i]
        z = np.load(self.files[i])
        rec = {k: z[k] for k in ("depth", "normal", "curv", "mask", "px", "vis",
                                 "landmarks", "extent", "view_basis", "view_target")}
        if len(self._d) >= self.cap:
            self._d.pop(next(iter(self._d)))
        self._d[i] = rec
        return rec


def make_input(rec, vi):
    """(5, H, W) float32: depth, normal(3), curvature -- normalised."""
    m = rec["mask"][vi]
    d = rec["depth"][vi].astype(np.float32)
    d[~m] = 0.0
    if m.any():
        # centre depth on the visible surface so the network sees relief, not
        # the absolute standoff distance (which is a constant per view anyway)
        d[m] = d[m] - np.median(d[m])
    d = np.clip(d / (float(rec["extent"]) * 0.5), -1.0, 1.0)

    n = rec["normal"][vi].astype(np.float32)
    n[~m] = 0.0
    c = np.clip(rec["curv"][vi].astype(np.float32), -3.0, 3.0) / 3.0
    c[~m] = 0.0
    return np.concatenate([d[None], n.transpose(2, 0, 1), c[None]], axis=0)


def build_index(cache, ear_ids):
    """(ear, view) pairs where the landmark is visible -- the training samples."""
    out = []
    for e in ear_ids:
        vis = cache.get(e)["vis"][:, LM]
        out += [(e, vi) for vi in np.flatnonzero(vis)]
    return out


def train(cache, train_ears, args, device):
    idx = build_index(cache, train_ears)
    print(f"  training samples: {len(idx)} (ear,view) pairs from {len(train_ears)} ears")
    net = HeatmapNet(n_landmarks=1, in_channels=5,
                     pretrained=args.weights if args.pretrained else None).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)

    for ep in range(args.epochs):
        net.train()
        rng.shuffle(idx)
        tot = n = 0.0
        for s in range(0, len(idx), args.batch):
            chunk = idx[s:s + args.batch]
            xb, tb = [], []
            for e, vi in chunk:
                rec = cache.get(e)
                xb.append(make_input(rec, vi))
                tb.append(rec["px"][vi, LM] / 4.0)      # heatmap is 1/4 resolution
            x = torch.from_numpy(np.stack(xb)).to(device)
            t = torch.from_numpy(np.stack(tb).astype(np.float32)).to(device)

            logits = net(x)
            H, W = logits.shape[-2:]
            tgt = gaussian_target(t[:, None, 0], t[:, None, 1], H, W,
                                  sigma=args.sigma, device=device)
            kl = F.kl_div(F.log_softmax(logits.reshape(len(chunk), 1, -1), dim=-1),
                          tgt.reshape(len(chunk), 1, -1), reduction="none").sum(-1).mean()
            pos, _ = soft_argmax(logits)
            l1 = (pos[:, 0] - t).abs().sum(-1).mean()
            loss = kl + args.pos_weight * l1

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(chunk)
            n += len(chunk)
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"    epoch {ep:3d}  loss {tot/max(n,1):.4f}", flush=True)
    return net.eval()


@torch.no_grad()
def predict_ear(net, rec, device, reject_mm: float = 4.0):
    """Fuse per-view 2D predictions into one 3D point."""
    pts, wts, per_view = [], [], []
    vis = rec["vis"][:, LM]
    for vi in np.flatnonzero(vis):
        x = torch.from_numpy(make_input(rec, vi)[None]).to(device)
        logits = net(x)
        pos, prob = soft_argmax(logits)
        px4, py4 = pos[0, 0].tolist()
        conf = float(prob.max().item())
        px, py = px4 * 4.0, py4 * 4.0                   # heatmap -> image coords

        res = rec["depth"].shape[-1]
        c, r = int(round(px)), int(round(py))
        if not (0 <= c < res and 0 <= r < res):
            continue
        # read depth near the prediction; step out if it landed on a hole
        d = None
        for rad in (0, 1, 2, 3):
            sl = np.s_[max(r - rad, 0):r + rad + 1, max(c - rad, 0):c + rad + 1]
            mm = rec["mask"][vi][sl]
            if mm.any():
                d = float(np.median(rec["depth"][vi][sl][mm].astype(np.float32)))
                break
        if d is None:
            continue

        rb, ub, fb = rec["view_basis"][vi]
        tgt = rec["view_target"][vi]
        ext = float(rec["extent"])
        X = ((px / (res - 1)) * 2.0 - 1.0) * ext
        Y = ((0.5 - py / (res - 1)) * 2.0) * ext
        p3 = tgt + X * rb + Y * ub + d * fb
        pts.append(p3)
        wts.append(conf)
        per_view.append(p3)

    if not pts:
        return None, []
    P = np.array(pts)
    W = np.array(wts)
    med = np.median(P, axis=0)
    keep = np.linalg.norm(P - med, axis=1) <= reject_mm
    if keep.sum() >= 1:
        P, W = P[keep], W[keep]
    return (P * W[:, None]).sum(0) / max(W.sum(), 1e-9), per_view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/views")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--max-folds", type=int, default=None, help="stop after N folds")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--pos-weight", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-cap", type=int, default=420,
                    help="ears held in RAM (~6.5MB each). Must exceed the "
                         "training-fold size or the cache thrashes and every "
                         "epoch re-reads from disk.")
    ap.add_argument("--weights", default="weights/resnet18.pth")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                    help="ablation: train from scratch, to isolate what ImageNet buys")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(Path(args.cache).glob("*.npz"))
    if not files:
        raise SystemExit(f"no cached views in {args.cache} -- run scripts/render_dataset.py")
    cache = EarCache(files, cap=args.cache_cap)

    # subject-wise split: left/right of one subject are the same anatomy
    subj = np.array([f.name.split("_")[0] for f in files])
    uniq = np.unique(subj)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, args.folds)
    print(f"{len(files)} ears / {len(uniq)} subjects, {args.folds} subject-wise folds, "
          f"device={device}, pretrained={args.pretrained}")

    errs, singles, nviews = [], [], []
    for fi, test_subj in enumerate(folds):
        if args.max_folds and fi >= args.max_folds:
            break
        te = np.flatnonzero(np.isin(subj, test_subj))
        tr = np.flatnonzero(~np.isin(subj, test_subj))
        print(f"\n=== fold {fi+1}/{len(folds)}: {len(tr)} train / {len(te)} test ears ===")
        t0 = time.time()
        net = train(cache, tr, args, device)

        for e in te:
            rec = cache.get(e)
            gt = rec["landmarks"][LM].astype(float)
            p, per_view = predict_ear(net, rec, device)
            if p is None:
                continue
            errs.append(float(np.linalg.norm(p - gt)))
            nviews.append(len(per_view))
            if per_view:
                singles.append(min(float(np.linalg.norm(q - gt)) for q in per_view))
        e = np.array(errs)
        print(f"  fold done in {(time.time()-t0)/60:.1f}min | running mean "
              f"{e.mean():.3f}mm median {np.median(e):.3f} n={len(e)}", flush=True)

    e = np.array(errs)
    s = np.array(singles)
    print("\n" + "=" * 74)
    print(f"LANDMARK 0, multi-view fused, {len(e)} held-out ears")
    print(f"  mean   {e.mean():.3f} mm")
    print(f"  median {np.median(e):.3f} mm")
    print(f"  p90    {np.percentile(e,90):.3f} mm")
    print(f"  best   {e.min():.3f}   worst {e.max():.3f}")
    print(f"  views fused per ear: mean {np.mean(nviews):.1f}")
    print(f"  best SINGLE view (oracle over views): mean {s.mean():.3f} mm")
    print("\ncomparators for the same landmark:")
    print("  0.285  rendering floor (256px, perfect prediction)")
    print("  0.688  curvature zero-crossing oracle")
    print("  2.09   current pipeline, outer_helix average")
    print("  6.717  crest walk")
    print(" 21.26   predict nothing, return landmark 6")


if __name__ == "__main__":
    main()
