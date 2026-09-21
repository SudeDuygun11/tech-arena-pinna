"""Elastic along-curve correspondence: 1-D multi-atlas for anchor slides.

Established: moving each anchor to its exact position ALONG OUR OWN predicted
curves gives 0.8795mm (t = -34) -- the whole remaining problem is predicting a
one-number slide per anchor. A ridge on a ~600-1000-dim along-curve profile is
a crude estimator with 267 training ears.

The principled tool for "which point along this curve corresponds to that one"
is curve registration: warping functions in functional data analysis (Ramsay &
Silverman 2005) and optimal re-parameterisation in elastic shape analysis
(Srivastava et al. 2011), both used for anatomical outlines. It is
non-parametric -- nothing to overfit -- and uses the whole curve.

METHOD, per test ear and contour (no ground truth at test time)
  1. candidates: the K training ears whose predicted anchor configuration is
     closest after rigid alignment (multi-atlas selection)
  2. align each candidate's predicted ear rigidly onto the test ear (Kabsch on
     predicted anchors); resample both predicted contours to M points by arc
     length
  3. slope-constrained dynamic programming (DTW, steps 0/1/2 per row) on a cost
     of aligned position + tangent direction + turning rate, with free start and
     end within a band
  4. carry the candidate's TRUE anchor arc position (projected onto its own
     predicted curve) through the warp onto the test curve
  5. combine over candidates, weighted by alignment cost
  6. slide the anchor along the test curve; step size chosen on inner folds;
     re-interpolate; score end to end, fully nested
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from feature_probe_r3 import AI, STEPS, load_sorted, rebuild
from src.foundations.contours import CONTOUR_SPECS
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid, kabsch
from src.foundations.splits import k_fold_subject_split

DATA = ROOT / "2026 Munich Tech Arena - Datas"
M = 80
K = 15
BAND = 8
CONTOURS = list(CONTOUR_SPECS.items())
ANCH_BY_C = [[(k, gi) for k, gi in enumerate(AI) if spec["range"][0] <= gi < spec["range"][1]]
             for _, spec in CONTOURS]


def arc(poly):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(seg)])


def resample(poly, m=M):
    cum = arc(poly); t = np.linspace(0, cum[-1], m)
    return np.stack([np.interp(t, cum, poly[:, c]) for c in range(3)], 1), cum[-1]


def point_at(poly, s):
    """Point at arc length s along poly, extrapolating past the ends."""
    cum = arc(poly)
    if s < 0:
        t = poly[1] - poly[0]; return poly[0] + s * t / (np.linalg.norm(t) + 1e-12)
    if s > cum[-1]:
        t = poly[-1] - poly[-2]; return poly[-1] + (s - cum[-1]) * t / (np.linalg.norm(t) + 1e-12)
    return np.stack([np.interp(s, cum, poly[:, c]) for c in range(3)])


def proj_arc(poly, p, ext=12.0):
    """Arc position of the point on (extended) poly nearest p."""
    cum = arc(poly); s = np.arange(-ext, cum[-1] + ext, 0.1)
    pts = np.array([point_at(poly, x) for x in s[::5]])          # coarse
    j = np.argmin(np.linalg.norm(pts - p, axis=1)); s0 = s[::5][j]
    fine = np.arange(s0 - 0.6, s0 + 0.6, 0.02)
    pts = np.array([point_at(poly, x) for x in fine])
    return fine[np.argmin(np.linalg.norm(pts - p, axis=1))]


def descriptor(curve):
    tan = np.gradient(curve, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    turn = np.linalg.norm(np.gradient(tan, axis=0), axis=1)
    return curve, tan, turn


def dtw_batch(test_desc, cand_descs, w_tan=4.0, w_turn=20.0):
    """Slope-constrained DP of one test curve against K candidates at once.
    Returns, for each candidate, j_of_i (test row -> candidate index) and cost."""
    tc, tt, tr = test_desc
    C = np.stack([cd[0] for cd in cand_descs]); T = np.stack([cd[1] for cd in cand_descs])
    Rr = np.stack([cd[2] for cd in cand_descs])
    cost = (np.linalg.norm(tc[None, :, None, :] - C[:, None, :, :], axis=3) ** 2
            + w_tan * (1 - np.abs((tt[None, :, None, :] * T[:, None, :, :]).sum(3)))
            + w_turn * (tr[None, :, None] - Rr[:, None, :]) ** 2)          # (K, M, M)
    Kc = len(cand_descs); INF = 1e18
    D = np.full((Kc, M), INF); D[:, :BAND] = cost[:, 0, :BAND]
    back = np.zeros((Kc, M, M), np.int8)
    for i in range(1, M):
        prev = D
        s0 = prev
        s1 = np.concatenate([np.full((Kc, 1), INF), prev[:, :-1]], 1)
        s2 = np.concatenate([np.full((Kc, 2), INF), prev[:, :-2]], 1)
        stack = np.stack([s0, s1, s2], 2)
        arg = stack.argmin(2); back[:, i] = arg
        D = cost[:, i] + stack.min(2)
    end = M - BAND + np.argmin(D[:, M - BAND:], axis=1)
    total = D[np.arange(Kc), end]
    jofi = np.zeros((Kc, M), int)
    for c in range(Kc):
        j = end[c]
        for i in range(M - 1, -1, -1):
            jofi[c, i] = j
            j -= back[c, i, j]
            j = max(j, 0)
    return jofi, total / M


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    n = len(pred)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f
    PA = pred[:, AI]
    base = np.linalg.norm(pred - truth, axis=2)

    print("precomputing curves and true anchor arc positions...", flush=True)
    t0 = time.time()
    polys = [[pred[i, spec["range"][0]:spec["range"][1]] for _, spec in CONTOURS] for i in range(n)]
    s_pred = np.zeros((n, len(AI))); s_true = np.zeros((n, len(AI))); L = np.zeros((n, 4))
    for i in range(n):
        for ci, (_, spec) in enumerate(CONTOURS):
            cum = arc(polys[i][ci]); L[i, ci] = cum[-1]
            for k, gi in ANCH_BY_C[ci]:
                s_pred[i, k] = cum[gi - spec["range"][0]]
                s_true[i, k] = proj_arc(polys[i][ci], truth[i, gi])
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    def predict_slides(test_ids, pool_ids):
        """Predicted arc positions (as slide amounts) for test ears from a training pool."""
        out = np.zeros((len(test_ids), len(AI)))
        for row, i in enumerate(test_ids):
            dist = []
            for r in pool_ids:
                Rm, t = kabsch(PA[r], PA[i])
                dist.append(np.sqrt(((apply_rigid(PA[r], Rm, t) - PA[i]) ** 2).sum(1).mean()))
            cands = np.asarray(pool_ids)[np.argsort(dist)[:K]]
            rig = {r: kabsch(PA[r], PA[i]) for r in cands}
            for ci in range(4):
                tc, Lt = resample(polys[i][ci]); tdesc = descriptor(tc)
                cdescs, cLs = [], []
                for r in cands:
                    cc, Lc = resample(apply_rigid(polys[r][ci], *rig[r])); cdescs.append(descriptor(cc)); cLs.append(Lc)
                jofi, cost = dtw_batch(tdesc, cdescs)
                w = np.exp(-(cost - cost.min()) / (np.median(cost) - cost.min() + 1e-9)); w /= w.sum()
                for k, gi in ANCH_BY_C[ci]:
                    est = []
                    for c, r in enumerate(cands):
                        u = s_true[r, k] / cLs[c] * (M - 1)                 # candidate sample coordinate
                        uc = np.clip(u, 0, M - 1)
                        ii = np.interp(uc, jofi[c] + np.linspace(0, 1e-6, M), np.arange(M))   # invert monotone j(i)
                        s_est = ii / (M - 1) * Lt + (u - uc) / (M - 1) * cLs[c]
                        est.append(s_est)
                    out[row, k] = float(np.sum(w * np.array(est))) - s_pred[i, k]
        return out

    def slide(i, k, amount):
        gi = AI[k]
        ci = next(c for c, (_, sp) in enumerate(CONTOURS) if sp["range"][0] <= gi < sp["range"][1])
        return point_at(polys[i][ci], s_pred[i, k] + amount)

    final = PA.copy(); t0 = time.time()
    for f in range(3):
        tr = np.flatnonzero(fold_of != f); te = np.flatnonzero(fold_of == f)
        us = np.array(sorted(set(subj[tr]))); np.random.default_rng(f).shuffle(us)
        inner = [tr[np.isin(subj[tr], p)] for p in np.array_split(us, 2)]
        oof = np.zeros((len(tr), len(AI))); pos = {ii: j for j, ii in enumerate(tr)}
        for j in range(2):
            b, a = inner[j], inner[1 - j]
            pb = predict_slides(b, a)
            for ii, v in zip(b, pb):
                oof[pos[ii]] = v
        w = np.zeros(len(AI))
        for k in range(len(AI)):
            errs = [np.mean([np.linalg.norm(slide(ii, k, s_ * oof[pos[ii], k]) - truth[ii, AI[k]]) for ii in tr])
                    for s_ in STEPS]
            w[k] = STEPS[int(np.argmin(errs))]
        pt = predict_slides(te, tr)
        for row, ii in enumerate(te):
            for k in range(len(AI)):
                final[ii, k] = slide(ii, k, w[k] * pt[row, k])
        print(f"  fold {f+1} done ({(time.time()-t0)/60:.1f} min); steps "
              + " ".join(f"{gi}:{w[k]:.2f}" for k, gi in enumerate(AI)), flush=True)

    na = pred.copy(); na[:, AI] = final
    e = np.linalg.norm(rebuild(pred, na) - truth, axis=2); ear = e.mean(1)
    r4 = np.load(ROOT / "results" / "feature_probe_r4.npz", allow_pickle=True)
    t2 = pred.copy(); t2[:, AI] = r4["anch_T2"]
    t2_ear = np.linalg.norm(rebuild(pred, t2) - truth, axis=2).mean(1)
    d0, d1 = ear - base.mean(1), ear - t2_ear
    print(f"\nbaseline {base.mean():.4f}   T2 {t2_ear.mean():.4f}   ceiling (exact slides) 0.8795")
    print(f"ELASTIC along-curve correspondence  {e.mean():.4f}  vs base {d0.mean():+.4f} "
          f"(t {d0.mean()/(d0.std(ddof=1)/np.sqrt(n)):+.2f})  vs T2 {d1.mean():+.4f} "
          f"(t {d1.mean()/(d1.std(ddof=1)/np.sqrt(n)):+.2f})")
    print("per anchor:  " + "  ".join(f"{gi}:{base[:, gi].mean():.2f}->{e[:, gi].mean():.2f}" for gi in AI))
    np.savez_compressed(ROOT / "results" / "elastic_along_curve.npz", keys=keys, anchors=final)


if __name__ == "__main__":
    main()
