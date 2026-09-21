"""Per-ear reference selection by registration FIT (no truth), follow-up to atlas_selection_probe.

Probe 1 (raw registration, fold 0, 134 ears):
  mean of 11 typical refs 3.298 | consensus-selected 20 of 40 3.115 (-0.18) |
  ORACLE best 11 of 40 by true error 2.407 (-0.89).
Choosing by truth is optimistic (it picks lucky noise too), so the question is how
much of that a truth-free fit signal recovers. Signals per (ear, reference):
  rigid  mean nearest-surface distance after rigid ICP (does the pose/shape fit?)
  bend   mean displacement the TPS needed on the control points (how different
         the reference had to be bent -- a similar ear needs little bending)
  lmbend same, landmark rows only (the region that matters)
  post   mean nearest-surface distance after TPS
Fusions: top-k by signal (k 5/11/20), soft weights exp(-z/tau) on the z-scored
signal, and signal+consensus rank averaging.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

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
PRED_CACHE = ROOT / "results" / "atlas_k40_fold0.npz"
SIG_CACHE = ROOT / "results" / "atlas_k40_fold0_signals.npz"
AI = list(ANCHOR_INDICES)


def signals(ref, V, crop_center, crop_radius):
    d = np.linalg.norm(V - crop_center[None, :], axis=1)
    cropped = V[d <= crop_radius]
    if len(cropped) < 20:
        cropped = V[d <= crop_radius * 2]
    tree = cKDTree(cropped)
    R, t, rigid_source = _rigid_icp(ref.control_points, cropped)
    base = apply_rigid(ref.landmarks, R, t)
    spline, warped = _nonrigid_refine(rigid_source, cropped)
    pred = spline(base)
    n_lm = len(ref.landmarks)
    return pred, np.array([tree.query(rigid_source)[0].mean(),
                           np.linalg.norm(warped - rigid_source, axis=1).mean(),
                           np.linalg.norm(warped[:n_lm] - rigid_source[:n_lm], axis=1).mean(),
                           tree.query(warped)[0].mean()])


def main():
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    train, test = next(iter(k_fold_subject_split(ds.subject_ids[:200], 3, 0)))
    ears = [(s, side) for s in test for side in ("left", "right")]
    truth = {(ex.subject_id, ex.side): ex.landmarks for ex in iter_landmarks_only(ds, list(test))}
    T = np.array([truth[e] for e in ears])
    preds = np.load(PRED_CACHE)["preds"]

    if SIG_CACHE.exists():
        S = np.load(SIG_CACHE)["S"]
    else:
        t0 = time.time()
        template = build_template(ds, list(train), k_references=preds.shape[1])
        S = np.zeros((len(ears), preds.shape[1], 4))
        part = ROOT / "results" / "atlas_k40_fold0_signals_partial.npz"
        start = 0
        if part.exists():
            S = np.load(part)["S"]; start = int(np.load(part)["done"])
            print(f"  resuming at ear {start}", flush=True)
        for i, (s, side) in enumerate(ears):
            if i < start:
                continue
            V = np.asarray(load_canonical_mesh(ds, s, side).vertices)
            te = time.time()
            for r, ref in enumerate(template.references):
                tr = time.time()
                p, S[i, r] = signals(ref, V, template.global_crop_center, template.global_crop_radius)
                if time.time() - tr > 20:
                    print(f"  slow: ear {i} ({s} {side}) ref {r} took {time.time()-tr:.0f}s, {len(V)} verts", flush=True)
                if i == 0 and r < 3:
                    assert np.allclose(p, preds[i, r], atol=1e-6), "signal pass does not reproduce cached registration"
            del V; gc.collect()
            np.savez_compressed(part, S=S, done=i + 1)
            if time.time() - te > 30:
                print(f"  ear {i} ({s} {side}) took {time.time()-te:.0f}s", flush=True)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(ears)} ears ({(time.time()-t0)/60:.1f} min)", flush=True)
        np.savez_compressed(SIG_CACHE, S=S)

    err_ear = lambda P: np.linalg.norm(P - T, axis=2).mean(1)
    base = err_ear(preds[:, :11].mean(1))
    ref_err = np.linalg.norm(preds - T[:, None], axis=3).mean(2)       # (ears, refs)
    def rep(label, P):
        e = err_ear(P); d = e - base
        print(f"{label:40s} {e.mean():.4f}  vs current {d.mean():+.4f} (t {d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.1f})")
    print(f"current (11 typical refs): {base.mean():.4f}\n")
    print("how well each signal ranks references (mean per-ear Spearman with true error):")
    from scipy.stats import spearmanr
    med = np.median(preds, axis=1, keepdims=True)
    cons = np.linalg.norm(preds - med, axis=3).mean(2)
    names = ["rigid", "bend", "lmbend", "post"]
    allsig = {n: S[:, :, k] for k, n in enumerate(names)}; allsig["consensus"] = cons
    for n, sig in allsig.items():
        rho = np.mean([spearmanr(sig[i], ref_err[i])[0] for i in range(len(ears))])
        print(f"  {n:10s} rho {rho:+.3f}")
    print()
    pick = lambda score, k: np.take_along_axis(preds, np.argsort(score, 1)[:, :k, None, None], 1).mean(1)
    for n, sig in allsig.items():
        for k in (5, 11, 20):
            rep(f"top {k:2d} by {n}", pick(sig, k))
    rank = lambda x: np.argsort(np.argsort(x, 1), 1)
    for combo in (("bend", "consensus"), ("lmbend", "consensus"), ("rigid", "bend", "consensus")):
        score = sum(rank(allsig[c]) for c in combo)
        for k in (11, 20):
            rep(f"top {k:2d} by rank({'+'.join(combo)})", pick(score, k))
    for n in ("bend", "lmbend", "consensus"):
        z = (allsig[n] - allsig[n].mean(1, keepdims=True)) / (allsig[n].std(1, keepdims=True) + 1e-9)
        for tau in (0.5, 1.0):
            w = np.exp(-z / tau); w /= w.sum(1, keepdims=True)
            rep(f"soft weights {n} tau {tau}", (preds * w[:, :, None, None]).sum(1))


if __name__ == "__main__":
    main()
