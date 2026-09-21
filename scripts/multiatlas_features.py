"""Multi-atlas evidence: use the 11 reference registrations individually.

Round 4 found the best correction so far uses ALL 85 predicted points (T2 ridge,
fully nested: 1.3127mm, t = -7.59 vs baseline, -4.34 vs the anchors-only V1):
the interior points carry independent evidence about where anchors sit along
their contours. Registration throws away a large source of exactly that kind of
evidence -- register_ensemble warps 11 reference ears onto the target and keeps
only their plain mean.

Multi-atlas segmentation solved this long ago: weight each atlas per location by
how well it matches the target locally, accounting for correlated atlas errors
(joint label fusion; Wang & Yushkevich, IEEE TPAMI 2013). This script:

  1. re-registers every ear with its outer-fold template (same k=11 references,
     same constants as the confirmed runs), keeping all 11 warped configurations
     and each reference's fit to the target surface -- globally and within 5mm
     of every landmark (mean nearest-neighbour distance of the warped reference
     surface to the target scan)
  2. scores, fully nested:
       M0  T2 (best so far)
       M1  T2 + locally similarity-weighted multi-atlas fusion (85 pts)
       M2  T2 + per-anchor spread across the 11 references (uncertainty)
       M3  T2 + the best-fitting reference's anchor per landmark (candidate)
       M4  T2 + M1 + M2 + M3
     and reports registration-level accuracy of uniform vs weighted fusion.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from feature_probe_r2 import fit_predict
from feature_probe_r3 import AI, STEPS, load_sorted, rebuild
from src.foundations.canonical import load_canonical_mesh
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid
from src.foundations.splits import k_fold_subject_split
from src.registration import registration as R
from src.registration.template import build_template

DATA = ROOT / "2026 Munich Tech Arena - Datas"
CACHE = ROOT / "results" / "multiatlas_cache.npz"
K_REF = 11


def register_one(ref, V, crop_center, crop_radius):
    """R.register, but also returning the warped reference surface's fit."""
    dist = np.linalg.norm(V - crop_center[None, :], axis=1)
    cropped = V[dist <= crop_radius]
    if len(cropped) < 20:
        cropped = V[dist <= crop_radius * 2]
    Rm, t, rigid_source = R._rigid_icp(ref.control_points, cropped)
    base = apply_rigid(ref.landmarks, Rm, t)
    spline, _ = R._nonrigid_refine(rigid_source, cropped)
    lm = spline(base)
    warped_cp = spline(rigid_source)
    d, _ = cKDTree(cropped).query(warped_cp)
    local = np.array([d[np.linalg.norm(warped_cp - p, axis=1) <= 5.0].mean()
                      if (np.linalg.norm(warped_cp - p, axis=1) <= 5.0).any() else d.mean() for p in lm])
    return lm, float(d.mean()), local


def extract(subj, side, keys, fold_of):
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        if list(c["keys"]) == list(keys):
            return c["refs"], c["glob"], c["local"]
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    folds = list(k_fold_subject_split(ds.subject_ids[:200], 3, 0))
    n = len(keys)
    refs = np.zeros((n, K_REF, 85, 3)); glob = np.zeros((n, K_REF)); local = np.zeros((n, K_REF, 85))
    t0 = time.time(); done = 0
    for f, (train_ids, _) in enumerate(folds):
        tpl = build_template(ds, train_ids, k_references=K_REF)
        print(f"  fold {f+1}: template built from {len(train_ids)} subjects "
              f"({len(tpl.references)} references)", flush=True)
        for i in np.flatnonzero(fold_of == f):
            m = load_canonical_mesh(ds, subj[i], side[i])
            V = np.asarray(m.vertices, float)
            del m; gc.collect()
            for r, ref in enumerate(tpl.references):
                refs[i, r], glob[i, r], local[i, r] = register_one(
                    ref, V, tpl.global_crop_center, tpl.global_crop_radius)
            del V; gc.collect()
            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                print(f"    {done}/{n} ears  {el/60:.1f} min  (~{el/done*(n-done)/60:.0f} min left)", flush=True)
        del tpl; gc.collect()
    np.savez_compressed(CACHE, keys=keys, refs=refs, glob=glob, local=local)
    return refs, glob, local


def nested(PA_full, TA, subj, fold_of, extra):
    """Fully nested per-anchor ridge on T2 features (+ optional per-ear extra)."""
    pa = PA_full[:, AI]; n = len(pa); final = pa.copy()

    def model(tr_i, te_i):
        out = np.zeros((len(te_i), len(AI), 3))
        for k in range(len(AI)):
            def X(ii):
                base = (PA_full[ii] - pa[ii, k][:, None, :]).reshape(len(ii), -1)
                if extra is None:
                    return base
                ex = extra(ii, k)
                return np.hstack([base, ex])
            out[:, k] = pa[te_i, k] + fit_predict(X(tr_i), TA[tr_i, k] - pa[tr_i, k], subj[tr_i], X(te_i))
        return out

    for f in range(3):
        tr = np.flatnonzero(fold_of != f); te = np.flatnonzero(fold_of == f)
        us = np.array(sorted(set(subj[tr]))); np.random.default_rng(f).shuffle(us)
        inner = [tr[np.isin(subj[tr], p)] for p in np.array_split(us, 2)]
        oof = {}
        for j in range(2):
            for ii, v in zip(inner[j], model(inner[1 - j], inner[j])):
                oof[ii] = v
        step = np.array([oof[ii] for ii in tr]) - pa[tr]
        w = np.array([STEPS[int(np.argmin([np.linalg.norm(pa[tr, k] + s * step[:, k] - TA[tr, k], axis=1).mean()
                                            for s in STEPS]))] for k in range(len(AI))])
        final[te] = pa[te] + w[None, :, None] * (model(tr, te) - pa[te])
    return final


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    n = len(pred)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f

    print("per-reference registration (cached after first run)...", flush=True)
    refs, glob, local = extract(subj, side, keys, fold_of)

    # registration-level: uniform vs locally weighted fusion
    uni = refs.mean(1)
    tau = np.median(local)
    wts = np.exp(-local / tau); wts /= wts.sum(1, keepdims=True)           # (n, 11, 85)
    wfus = (refs * wts[..., None]).sum(1)
    e_u = np.linalg.norm(uni - truth, axis=2); e_w = np.linalg.norm(wfus - truth, axis=2)
    print(f"\nregistration prior only: uniform mean {e_u.mean():.4f}  locally weighted {e_w.mean():.4f}  "
          f"(anchors {e_u[:, AI].mean():.4f} -> {e_w[:, AI].mean():.4f})")
    spread = refs[:, :, AI].std(1).mean(-1)                                 # (n, 15)
    best_ref = np.argmin(local[:, :, AI], axis=1)                           # (n, 15)
    best_cand = np.stack([refs[np.arange(n), best_ref[:, k], AI[k]] for k in range(len(AI))], 1)

    PA, TA = pred[:, AI], truth[:, AI]
    blocks = {
        "fusion": lambda ii, k: (wfus[ii] - PA[ii, k][:, None, :]).reshape(len(ii), -1),
        "spread": lambda ii, k: spread[ii],
        "bestref": lambda ii, k: (best_cand[ii] - PA[ii, k][:, None, :]).reshape(len(ii), -1),
    }

    def combo(names):
        return lambda ii, k: np.hstack([blocks[nm](ii, k) for nm in names])

    variants = [("M0 T2 all-85 (best so far)", None),
                ("M1 + weighted multi-atlas fusion", combo(["fusion"])),
                ("M2 + reference spread", combo(["spread"])),
                ("M3 + best-fitting reference anchor", combo(["bestref"])),
                ("M4 + all multi-atlas blocks", combo(["fusion", "spread", "bestref"]))]
    base = np.linalg.norm(pred - truth, axis=2)
    print(f"\nbaseline {base.mean():.4f}")
    ref_ear = None
    for name, extra in variants:
        A = nested(pred, TA, subj, fold_of, extra)
        na = pred.copy(); na[:, AI] = A
        e = np.linalg.norm(rebuild(pred, na) - truth, axis=2); ear = e.mean(1)
        ref_ear = ear if ref_ear is None else ref_ear
        d0, d1 = ear - base.mean(1), ear - ref_ear
        print(f"  {name:36s} {e.mean():.4f}  vs base {d0.mean():+.4f} (t {d0.mean()/(d0.std(ddof=1)/np.sqrt(n)):+.2f})"
              f"  vs M0 {d1.mean():+.4f} (t {d1.mean()/(d1.std(ddof=1)/np.sqrt(n)+1e-12):+.2f})"
              f"   74 {e[:, 74].mean():.3f}  6 {e[:, 6].mean():.3f}  22 {e[:, 22].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
