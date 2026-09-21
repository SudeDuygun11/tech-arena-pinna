"""Round 4: can relational correction get past its linear ceiling?

Rounds 1-3 established that the only information locating the sliding anchors
is where the OTHER landmarks are, and that six LINEAR relational formulations
(ridge on other anchors, + organizer rules, + whole 85-point shape, + head
context, posterior shape model, per-anchor best) all saturate at ~1.318-1.325mm
(-0.045mm, t up to -7.1). Two ways past a linear ceiling are tested here:

  N   NONLINEAR relational models, as in 2025-2026 landmark work that refines
      predicted configurations with learned inter-landmark dependencies:
        N1  MLP, shared across anchors, input = predicted anchors (aligned)
        N2  MLP, input = all 85 predicted points (aligned)
        N3  gradient-boosted trees per anchor on the relational features
  S   SEQUENTIAL, clinician-style correction ("Tracing Like a Clinician",
      arXiv 2605.03358, 2026: easy reference landmarks inform dependent ones).
      Anchors are corrected in order of reliability, and each stage sees the
      CORRECTED positions of the stages before it:
        stage 1  junctions        46 54 50 25 24 42 0 75 84 33
        stage 2  cross/height ref 55 64
        stage 3  extrema          6 22
        stage 4  constructed      74

RIGOUR FIX vs round 3
In round 3 the per-anchor step size was chosen on training-fold ears whose
proposals came from models that had seen the test fold -- a small second-order
leak. Here everything is FULLY NESTED: for each outer fold, inner cross-fitting
on the training ears produces honest proposals for choosing the step, then a
model fitted on all training ears predicts the test fold. V1 is re-scored the
same way, so all comparisons in this file are apples to apples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from feature_probe_r2 import fit_predict
from feature_probe_r3 import AI, KIND, STEPS, load_sorted, rebuild
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid, kabsch
from src.foundations.splits import k_fold_subject_split

DATA = ROOT / "2026 Munich Tech Arena - Datas"
STAGES = [[46, 54, 50, 25, 24, 42, 0, 75, 84, 33], [55, 64], [6, 22], [74]]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ models
def ridge_rel(P_tr, T_tr, g_tr, P_te, k_idx, full_tr=None, full_te=None):
    """Per-anchor ridge on the other anchors (V1) -- optionally on all 85 points."""
    out = np.zeros((len(P_te), len(k_idx), 3))
    for j, k in enumerate(k_idx):
        src_tr = full_tr if full_tr is not None else P_tr
        src_te = full_te if full_te is not None else P_te
        Xtr = (src_tr - P_tr[:, k][:, None, :]).reshape(len(P_tr), -1)
        Xte = (src_te - P_te[:, k][:, None, :]).reshape(len(P_te), -1)
        out[:, j] = P_te[:, k] + fit_predict(Xtr, T_tr[:, k] - P_tr[:, k], g_tr, Xte)
    return out


def _align(P_anch, ref):
    Rs, ts = [], []
    for p in P_anch:
        R, t = kabsch(p, ref); Rs.append(R); ts.append(t)
    return np.array(Rs), np.array(ts)


def mlp_rel(P_tr, T_tr, g_tr, P_te, k_idx, full_tr=None, full_te=None, seeds=5):
    """Shared MLP: aligned configuration -> aligned corrections for anchors k_idx."""
    ref = P_tr.mean(0)
    for _ in range(3):
        R, t = _align(P_tr, ref)
        ref = np.mean([apply_rigid(P_tr[i], R[i], t[i]) for i in range(len(P_tr))], 0)
    Rtr, ttr = _align(P_tr, ref); Rte, tte = _align(P_te, ref)

    def feats(P, full, R, t):
        src = full if full is not None else P
        return np.array([apply_rigid(src[i], R[i], t[i]).ravel() for i in range(len(P))])

    Xtr, Xte = feats(P_tr, full_tr, Rtr, ttr), feats(P_te, full_te, Rte, tte)
    Ytr = np.array([((T_tr[i, k_idx] - P_tr[i, k_idx]) @ Rtr[i].T).ravel() for i in range(len(P_tr))])
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-6; my, sy = Ytr.mean(0), Ytr.std(0) + 1e-6
    Xtr_t = torch.tensor((Xtr - mx) / sx, dtype=torch.float32, device=DEV)
    Ytr_t = torch.tensor((Ytr - my) / sy, dtype=torch.float32, device=DEV)
    Xte_t = torch.tensor((Xte - mx) / sx, dtype=torch.float32, device=DEV)
    preds = []
    for seed in range(seeds):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        perm = rng.permutation(len(Xtr)); nv = max(8, len(Xtr) // 6)
        va, tr = perm[:nv], perm[nv:]
        net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(256, Ytr.shape[1])).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        best, best_state, bad = 1e9, None, 0
        for ep in range(600):
            net.train(); opt.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(net(Xtr_t[tr]), Ytr_t[tr]); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = torch.nn.functional.smooth_l1_loss(net(Xtr_t[va]), Ytr_t[va]).item()
            if vl < best - 1e-5:
                best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad > 40:
                    break
        net.load_state_dict(best_state); net.eval()
        with torch.no_grad():
            preds.append(net(Xte_t).cpu().numpy() * sy + my)
    Yhat = np.mean(preds, 0).reshape(len(P_te), len(k_idx), 3)
    return np.array([P_te[i, k_idx] + Yhat[i] @ Rte[i] for i in range(len(P_te))])


def gbt_rel(P_tr, T_tr, g_tr, P_te, k_idx, **_):
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = np.zeros((len(P_te), len(k_idx), 3))
    for j, k in enumerate(k_idx):
        Xtr = (P_tr - P_tr[:, k][:, None, :]).reshape(len(P_tr), -1)
        Xte = (P_te - P_te[:, k][:, None, :]).reshape(len(P_te), -1)
        for c in range(3):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=10, l2_regularization=1.0, random_state=0)
            m.fit(Xtr, T_tr[:, k, c] - P_tr[:, k, c])
            out[:, j, c] = P_te[:, k, c] + m.predict(Xte)
    return out


# ------------------------------------------------------------------ nested harness
def nested(model, pred, truth, subj, fold_of, sequential=False, use_full=False):
    """Returns corrected anchor array (n, 15, 3) with per-anchor steps chosen on
    INNER out-of-fold proposals only."""
    PA, TA = pred[:, AI], truth[:, AI]
    pos = {gi: k for k, gi in enumerate(AI)}
    order = STAGES if sequential else [AI]
    final = PA.copy()
    for f in range(3):
        tr = np.flatnonzero(fold_of != f); te = np.flatnonzero(fold_of == f)
        # inner split of training subjects
        us = np.array(sorted(set(subj[tr]))); np.random.default_rng(f).shuffle(us)
        inner = [np.flatnonzero(np.isin(subj[tr], part)) for part in np.array_split(us, 2)]

        def run(tr_i, te_i, cur_tr, cur_te):
            """One stage-by-stage pass; returns proposals for te_i (full step)."""
            prop_te = cur_te.copy(); cur_tr = cur_tr.copy()
            for si, stage in enumerate(order):
                ks = [pos[g] for g in stage]
                kw = dict(full_tr=pred[tr_i] if use_full else None, full_te=pred[te_i] if use_full else None)
                p_te = model(cur_tr, TA[tr_i], subj[tr_i], prop_te, ks, **kw)
                if sequential:
                    # training ears need honest corrected positions for the next stage:
                    # 2-way cross-fit inside the training set
                    half = np.array_split(np.random.default_rng(100 + si).permutation(len(tr_i)), 2)
                    p_tr = cur_tr.copy()
                    for h in range(2):
                        a, b = half[h], half[1 - h]
                        kwh = dict(full_tr=pred[tr_i][a] if use_full else None,
                                   full_te=pred[tr_i][b] if use_full else None)
                        p_tr[np.ix_(b, ks)] = model(cur_tr[a], TA[tr_i][a], subj[tr_i][a], cur_tr[b], ks, **kwh)
                    cur_tr[:, ks] = p_tr[:, ks]
                prop_te[:, ks] = p_te
            return prop_te

        # inner out-of-fold proposals on training ears -> choose steps
        oof = PA[tr].copy()
        for j in range(2):
            a, b = inner[1 - j], inner[j]
            oof[b] = run(tr[a], tr[b], PA[tr[a]], PA[tr[b]])
        step_tr = oof - PA[tr]
        w = np.zeros(len(AI))
        for k in range(len(AI)):
            errs = [np.linalg.norm(PA[tr, k] + s * step_tr[:, k] - TA[tr, k], axis=1).mean() for s in STEPS]
            w[k] = STEPS[int(np.argmin(errs))]
        prop = run(tr, te, PA[tr], PA[te])
        final[te] = PA[te] + w[None, :, None] * (prop - PA[te])
    return final


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    n = len(pred)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f
    base = np.linalg.norm(pred - truth, axis=2)

    variants = [
        ("V1 ridge (nested)",            dict(model=ridge_rel)),
        ("S1 sequential ridge",          dict(model=ridge_rel, sequential=True)),
        ("N1 MLP anchors",               dict(model=mlp_rel)),
        ("N3 gradient-boosted trees",    dict(model=gbt_rel)),
        ("T2 ridge all-85 (nested)",     dict(model=ridge_rel, use_full=True)),
        ("N2 MLP all-85",                dict(model=mlp_rel, use_full=True)),
        ("S2 sequential MLP",            dict(model=mlp_rel, sequential=True)),
    ]
    print(f"baseline {base.mean():.4f}\n")
    print(f"{'variant':30s} {'overall':>8s} {'vs base':>8s} {'t':>7s} {'vs V1':>8s} {'t':>7s}   "
          f"{'74':>6s} {'6':>6s} {'22':>6s} {'64':>6s} {'55':>6s}")
    ears, anch = {}, {}
    for name, kw in variants:
        A = nested(pred=pred, truth=truth, subj=subj, fold_of=fold_of, **kw)
        na = pred.copy(); na[:, AI] = A
        e = np.linalg.norm(rebuild(pred, na) - truth, axis=2)
        ears[name] = e.mean(1); anch[name] = A
        d0 = ears[name] - base.mean(1)
        v1 = ears["V1 ridge (nested)"]; d1 = ears[name] - v1
        t1 = d1.mean() / (d1.std(ddof=1) / np.sqrt(n) + 1e-12)
        print(f"{name:30s} {e.mean():8.4f} {d0.mean():+8.4f} {d0.mean()/(d0.std(ddof=1)/np.sqrt(n)):+7.2f} "
              f"{d1.mean():+8.4f} {t1:+7.2f}   {e[:, 74].mean():6.3f} {e[:, 6].mean():6.3f} "
              f"{e[:, 22].mean():6.3f} {e[:, 64].mean():6.3f} {e[:, 55].mean():6.3f}", flush=True)
    np.savez_compressed(ROOT / "results" / "feature_probe_r4.npz", keys=keys,
                        **{f"anch_{nm.split()[0]}": v for nm, v in anch.items()})
    print("\nsaved -> results/feature_probe_r4.npz")


if __name__ == "__main__":
    main()
