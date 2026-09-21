"""Push reference selection further: bigger pool, LOCAL fit per landmark, learned selector.

Registration-only probe on fold 0 (134 held-out ears), no networks. Established:
  11 typical refs, same for every ear             3.298
  best 5 of 40 by global surface fit (no truth)   2.741
  best 11 of 40 chosen by TRUE error (ceiling)    2.407
In the pipeline (Wing+aug+SWA, 3 seeds) selection took 1.630 -> 1.586.

Here, per (ear, reference), we store the warped prediction, the 4 global fit
signals, and a LOCAL fit per landmark: mean nearest-surface distance of the
warped reference's surface points within 8mm of that landmark. Then score:
  G  global top-k by fit, pool 40 vs 100
  L  per-landmark top-k by local fit
  M  mix of local + global rank
  S  learned selector (HistGradientBoosting, 5-fold CV grouped by SUBJECT so an
     ear's twin never sits in train and test) predicting each reference's error
     at each landmark from: local fit, global fits, consensus deviation, typicality
  O  oracles: best k by true error, global and per landmark (ceilings)
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid
from src.foundations.splits import k_fold_subject_split
from src.registration.registration import _nonrigid_refine, _rigid_icp
from src.registration.template import build_template

DATA = ROOT / "2026 Munich Tech Arena - Datas"
POOL = 100
RADIUS = 8.0
CACHE = ROOT / "results" / "atlas_pool100_fold0.npz"
PART = ROOT / "results" / "atlas_pool100_fold0_partial.npz"
AI = list(ANCHOR_INDICES)


def one(ref, V, cc, cr):
    d = np.linalg.norm(V - cc[None, :], axis=1)
    cropped = V[d <= cr]
    if len(cropped) < 20:
        cropped = V[d <= cr * 2]
    tree = cKDTree(cropped)
    R, t, rigid = _rigid_icp(ref.control_points, cropped)
    base = apply_rigid(ref.landmarks, R, t)
    spline, warped = _nonrigid_refine(rigid, cropped)
    pred = spline(base)
    n = len(ref.landmarks)
    dw = tree.query(warped)[0]
    S = np.array([tree.query(rigid)[0].mean(), np.linalg.norm(warped - rigid, axis=1).mean(),
                  np.linalg.norm(warped[:n] - rigid[:n], axis=1).mean(), dw.mean()])
    surf, dsurf = warped[n:], dw[n:]
    m = cdist(pred, surf) <= RADIUS
    cnt = m.sum(1)
    loc = np.where(cnt > 0, (m * dsurf[None]).sum(1) / np.maximum(cnt, 1), dw.mean())
    return pred, S, loc


def collect(ds, train, ears):
    E = len(ears)
    if CACHE.exists():
        c = np.load(CACHE); return c["preds"], c["S"], c["LOC"]
    t0 = time.time()
    template = build_template(ds, list(train), k_references=POOL)
    print(f"template {len(template.references)} refs in {time.time()-t0:.0f}s", flush=True)
    preds = np.zeros((E, POOL, 85, 3), np.float32); S = np.zeros((E, POOL, 4)); LOC = np.zeros((E, POOL, 85), np.float32)
    start = 0
    if PART.exists():
        c = np.load(PART); preds, S, LOC, start = c["preds"], c["S"], c["LOC"], int(c["done"])
        print(f"resuming at ear {start}", flush=True)
    old = ROOT / "results" / "atlas_k40_fold0.npz"
    for i, (s, side) in enumerate(ears):
        if i < start:
            continue
        V = np.asarray(load_canonical_mesh(ds, s, side).vertices)
        te = time.time()
        for r, ref in enumerate(template.references):
            p, S[i, r], LOC[i, r] = one(ref, V, template.global_crop_center, template.global_crop_radius)
            preds[i, r] = p
            if i == 0 and r < 3 and old.exists():
                ok = np.allclose(p, np.load(old)["preds"][0, r], atol=1e-3)
                print(f"  check ear 0 ref {r} reproduces earlier probe: {ok}", flush=True)
        del V; gc.collect()
        np.savez(PART, preds=preds, S=S, LOC=LOC, done=i + 1)
        if time.time() - te > 90:
            print(f"  slow ear {i}: {time.time()-te:.0f}s", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{E} ears ({(time.time()-t0)/60:.1f} min)", flush=True)
    np.savez_compressed(CACHE, preds=preds, S=S, LOC=LOC)
    return preds, S, LOC


def main():
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    train, test = next(iter(k_fold_subject_split(ds.subject_ids[:200], 3, 0)))
    ears = [(s, side) for s in test for side in ("left", "right")]
    truth = {(ex.subject_id, ex.side): ex.landmarks for ex in iter_landmarks_only(ds, list(test))}
    T = np.array([truth[e] for e in ears]); subj = np.array([s for s, _ in ears])
    preds, S, LOC = collect(ds, train, ears)
    E, R = preds.shape[:2]
    preds = preds.astype(np.float64)

    def rep(label, P, ref=None):
        e = np.linalg.norm(P - T, axis=2)
        line = f"{label:44s} all {e.mean():.4f}  anchors {e[:, AI].mean():.4f}"
        if ref is not None:
            d = e.mean(1) - ref
            line += f"  vs base {d.mean():+.4f} (t {d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.1f})"
        print(line, flush=True)
        return e.mean(1)

    base = rep("11 typical refs (current)", preds[:, :11].mean(1))
    pick_g = lambda score, k, n=R: np.take_along_axis(preds[:, :n], np.argsort(score[:, :n], 1)[:, :k, None, None], 1).mean(1)
    print("\nG: global fit (post) top-k")
    rep("pool 40, top 5  (earlier finding)", pick_g(S[:, :, 3], 5, 40), base)
    for k in (3, 5, 8, 12):
        rep(f"pool {R}, top {k}", pick_g(S[:, :, 3], k), base)

    def pick_local(score, k):                       # score (E, R, 85): per landmark
        idx = np.argsort(score, 1)[:, :k, :]         # (E, k, 85)
        return np.take_along_axis(preds, idx[..., None], 1).mean(1)
    print("\nL: local fit per landmark, top-k")
    for k in (3, 5, 8, 12):
        rep(f"pool {R}, top {k}", pick_local(LOC, k), base)

    rk = lambda x, ax: np.argsort(np.argsort(x, ax), ax)
    gpost = np.repeat(S[:, :, 3][:, :, None], 85, 2)
    print("\nM: rank mix (local + global)")
    for k in (5, 8):
        rep(f"pool {R}, top {k}, rank(loc)+rank(post)", pick_local(rk(LOC, 1) + rk(gpost, 1), k), base)

    print("\nO: oracles (use truth: ceilings only)")
    err = np.linalg.norm(preds - T[:, None], axis=3)                     # (E, R, 85)
    for k in (5, 11):
        idx = np.argsort(err.mean(2), 1)[:, :k]
        rep(f"best {k} per ear by true error", np.take_along_axis(preds, idx[:, :, None, None], 1).mean(1), base)
        rep(f"best {k} per LANDMARK by true error", pick_local(err, k), base)

    print("\nS: learned selector (grouped CV by subject)")
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    z = lambda x, ax: (x - x.mean(ax, keepdims=True)) / (x.std(ax, keepdims=True) + 1e-9)
    med = np.median(preds, axis=1, keepdims=True)
    dev = np.linalg.norm(preds - med, axis=3)                            # (E, R, 85)
    feats = np.stack([z(LOC, 1), LOC, z(dev, 1), dev,
                      np.repeat(z(S[:, :, 0], 1)[:, :, None], 85, 2), np.repeat(z(S[:, :, 1], 1)[:, :, None], 85, 2),
                      np.repeat(z(S[:, :, 3], 1)[:, :, None], 85, 2), np.repeat(S[:, :, 3][:, :, None], 85, 2),
                      np.broadcast_to(np.arange(R)[None, :, None], (E, R, 85)).astype(float),
                      np.broadcast_to(np.arange(85)[None, None, :], (E, R, 85)).astype(float)], -1).astype(np.float32)
    pred_err = np.zeros((E, R, 85), np.float32)
    rng = np.random.default_rng(0)
    for f, (tr, te) in enumerate(GroupKFold(5).split(np.arange(E), groups=subj)):
        Xtr = feats[tr].reshape(-1, feats.shape[-1]); ytr = err[tr].reshape(-1)
        sel = rng.choice(len(Xtr), min(400000, len(Xtr)), replace=False)
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=0)
        m.fit(Xtr[sel], ytr[sel])
        pred_err[te] = m.predict(feats[te].reshape(-1, feats.shape[-1])).reshape(len(te), R, 85)
        print(f"  fold {f} done", flush=True)
    for k in (3, 5, 8, 12):
        rep(f"learned selector, top {k}", pick_local(pred_err, k), base)
    print("\nper part, best rows:")
    from src.foundations.contours import CONTOUR_SPECS
    cand = {"11 typical": preds[:, :11].mean(1), "G pool40 top5": pick_g(S[:, :, 3], 5, 40),
            f"L pool{R} top5": pick_local(LOC, 5), "S learned top5": pick_local(pred_err, 5)}
    for name, spec in CONTOUR_SPECS.items():
        lo, hi = spec["range"]
        print(f"  {name:20s} " + "  ".join(f"{k} {np.linalg.norm(P - T, axis=2)[:, lo:hi].mean():.3f}" for k, P in cand.items()))


if __name__ == "__main__":
    main()
