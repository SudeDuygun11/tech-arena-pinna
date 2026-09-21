"""Feature probe, round 2: literature feature families never tried on this data.

Round 1 (scripts/feature_probe.py) found that no feature ALREADY in the codebase
carries along-contour information: per-anchor feature choice improved the along
residual only 0.835 -> 0.828, and landmark 74 was unrecoverable by any set. It
also found our 'curvature' channel adds almost nothing -- but that channel is a
one-ring Laplacian, previously measured to be scanner noise at this resolution.
So round 1 could not distinguish "curvature is uninformative" from "our
curvature estimator is bad". Round 2 tests families from the ear, craniofacial
and shape-analysis literature:

  R1 shape index + curvedness at 2mm and 3mm scale
       Koenderink & van Doorn 1992; used for 3D ear helix/antihelix detection
       (Chen & Bhanu, IEEE TPAMI 2007) and 16 ear fiducials (Lei, You &
       Abdel-Mottaleb, IEEE T-SMC 2016).
  R2 ridge DIRECTION (principal-direction orientation tensor) and how fast it
       TURNS. Motivation: an extremum anchor like 6 or 22 ("largest extent")
       sits where the helix ridge turns fastest, so turning rate should peak
       exactly at our worst sliders.
  R3 histogram descriptors: spin image (Johnson & Hebert 1999) and the Local
       Surface Patch 2D histogram of shape index vs normal angle (Chen & Bhanu
       2007). Both are rotation-invariant about the normal BY DESIGN, so they
       are expected to be weak on the along/across distinction -- included as
       the literature baseline, not as a favourite.
  R4 Heat Kernel Signature (Sun, Ovsjanikov & Guibas 2009) and Wave Kernel
       Signature (Aubry et al. 2011); used for craniofacial landmark detection.
       Intrinsic and multi-scale: at long diffusion times a point's value
       depends on the whole ear, so it is the one family NOT limited to the
       patch -- the only candidate that could locate landmark 74.
  R5 position relative to the OTHER predicted anchors. Two versions:
       (a) inside the probe, using the real saved predictions for the other
           anchors (flattering: probe displacements are independent of the
           other anchors' errors, real pipeline errors are not);
       (b) HONEST: directly on the saved predictions, a cross-fitted ridge from
           the predicted anchor configuration to each anchor's true correction,
           scored end to end.
  R6 depth from the ear plane (SMC2024 annotates pinna landmarks on depth
       images; concha depth is acoustically important) and fold ENCLOSURE
       (fraction of nearby surface lying above the tangent plane -- separates
       the exposed helix rim from the inside of the fold).

APPROXIMATIONS, stated up front
  * principal curvatures from a batched quadric fit on fixed-k neighbourhoods
  * HKS/WKS from a point-cloud graph Laplacian on a ~1.2mm voxel subsample, not
    a cotangent mesh Laplacian -- adequate to measure information content
  * geodesic distances in R5 replaced by Euclidean
  * crops centred on the ground-truth landmark centroid, as in round 1

Same guards as round 1: beat 'no correction'; z > 3 vs the base set, paired
over subjects; per-anchor choice CROSS-FITTED on the other folds.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.correction.patch_features import _local_curvature
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split
from src.interpolation.interpolation import resample_uniform

G = 4                 # descriptor grid per axis (64 cells)
GH = 3                # coarser grid for HKS/WKS (27 cells x 8 scales)
R = 7.0               # patch radius, as in the pipeline
AI = list(ANCHOR_INDICES)
ANCHOR_KIND = {0: "junction", 6: "extremum", 22: "extremum", 24: "junction",
               25: "junction", 33: "junction", 42: "junction", 46: "junction",
               50: "junction", 54: "junction", 55: "height-ref", 64: "cross-ref",
               74: "constructed", 75: "junction", 84: "junction"}

FEATURE_SETS = {
    "B_base":        ["occ", "nrm", "curv"],
    "R1_shape":      ["occ", "nrm", "curv", "si2", "cv2", "si3", "cv3"],
    "R2_ridge_dir":  ["occ", "nrm", "curv", "ori", "turn"],
    "R3_histograms": ["occ", "nrm", "curv", "spin", "lsp"],
    "R4_heat":       ["occ", "nrm", "curv", "hks", "wks"],
    "R5_relational": ["occ", "nrm", "curv", "rel"],
    "R6_depth_fold": ["occ", "nrm", "curv", "depth", "encl"],
    "R4R5_heat_rel": ["occ", "nrm", "curv", "hks", "wks", "rel"],
    "ALL_local":     ["occ", "nrm", "curv", "si2", "cv2", "si3", "cv3", "ori", "turn",
                      "spin", "lsp", "depth", "encl"],
    "ALL":           ["occ", "nrm", "curv", "si2", "cv2", "si3", "cv3", "ori", "turn",
                      "spin", "lsp", "hks", "wks", "rel", "depth", "encl"],
}


# ----------------------------------------------------------------- geometry
def principal_curvatures(V, N, tree, idx, k):
    """Batched quadric fit z = a x^2 + b xy + c y^2 + d x + e y + f in each
    vertex's tangent frame over its k nearest neighbours. Returns k1 >= k2 and
    the 3D direction of the larger-magnitude principal curvature."""
    _, nb = tree.query(V[idx], k=k)
    n = N[idx]
    ref = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    t1 = ref - (ref * n).sum(1, keepdims=True) * n
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    P = V[nb] - V[idx][:, None, :]
    x = (P * t1[:, None, :]).sum(2); y = (P * t2[:, None, :]).sum(2); z = (P * n[:, None, :]).sum(2)
    A = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=2)
    AtA = np.einsum("nki,nkj->nij", A, A) + 1e-9 * np.eye(6)
    Atz = np.einsum("nki,nk->ni", A, z)
    coef = np.linalg.solve(AtA, Atz[..., None])[..., 0]
    H = np.stack([np.stack([2 * coef[:, 0], coef[:, 1]], 1),
                  np.stack([coef[:, 1], 2 * coef[:, 2]], 1)], 1)
    w, U = np.linalg.eigh(H)                               # ascending
    k2, k1 = w[:, 0], w[:, 1]
    big = np.where(np.abs(k1) >= np.abs(k2), 1, 0)
    u = U[np.arange(len(idx)), :, big]                    # 2D direction
    d = u[:, :1] * t1 + u[:, 1:2] * t2
    return k1, k2, d / np.linalg.norm(d, axis=1, keepdims=True)


def shape_index(k1, k2):
    return (2.0 / np.pi) * np.arctan2(k1 + k2, np.abs(k1 - k2) + 1e-12)


def graph_hks_wks(V, n_eig=60, voxel=1.2, n_scales=8):
    """HKS and WKS on a voxel-subsampled point cloud via a Gaussian-weighted kNN
    graph Laplacian (Belkin-Niyogi). Returns per-vertex (n_scales,) arrays for
    every ORIGINAL vertex by nearest-subsample lookup."""
    key = np.floor(V / voxel).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    S = V[first]
    kk = 10
    t = cKDTree(S)
    dist, nb = t.query(S, k=kk + 1)
    h = np.mean(dist[:, 1:] ** 2)
    rows = np.repeat(np.arange(len(S)), kk); cols = nb[:, 1:].ravel()
    w = np.exp(-dist[:, 1:].ravel() ** 2 / (4 * h))
    W = coo_matrix((w, (rows, cols)), shape=(len(S), len(S))).tocsr()
    W = 0.5 * (W + W.T)
    Dg = np.asarray(W.sum(1)).ravel()
    L = diags(Dg) - W
    lam, phi = eigsh(L, k=min(n_eig, len(S) - 2), M=diags(Dg), sigma=-1e-6, which="LM")
    order = np.argsort(lam); lam = lam[order]; phi = phi[:, order]
    # Drop EVERY near-zero mode, not only the first. A kNN graph that falls apart
    # into several pieces has one zero eigenvalue per piece; a zero left in lam_pos
    # makes tmax infinite and log(lam) = -inf, which turned the whole R4 set into
    # NaN in the first full run (a single NaN sample poisons that set's regression).
    keep = lam > 1e-8 * max(float(lam.max()), 1e-12)
    lam_pos = lam[keep]; phi2 = phi[:, keep] ** 2
    tmin, tmax = 4 * np.log(10) / lam_pos.max(), 4 * np.log(10) / lam_pos.min()
    ts = np.geomspace(tmin, tmax, n_scales)
    hks = np.stack([(phi2 * np.exp(-lam_pos * tt)).sum(1) / np.exp(-lam_pos * tt).sum()
                    for tt in ts], 1)
    le = np.log(lam_pos + 1e-12)
    es = np.linspace(le.min(), le.max(), n_scales + 2)[1:-1]
    sig = 7 * (es[1] - es[0])
    wks = np.stack([(phi2 * np.exp(-(e - le) ** 2 / (2 * sig ** 2))).sum(1)
                    / (np.exp(-(e - le) ** 2 / (2 * sig ** 2)).sum() + 1e-12) for e in es], 1)
    nearest = t.query(V)[1]
    hks = hks[nearest]; wks = wks[nearest]
    return hks / (hks.mean(0) + 1e-12), wks / (wks.mean(0) + 1e-12)


def grid_means(rel, vals, g):
    """Per-cell mean of per-point values (n,) or (n,c) over a g^3 grid in [-1,1]^3."""
    cell3 = np.clip(((rel + 1.0) * 0.5 * g).astype(int), 0, g - 1)
    cell = cell3[:, 0] * g * g + cell3[:, 1] * g + cell3[:, 2]
    cnt = np.maximum(np.bincount(cell, minlength=g ** 3), 1)
    vals = vals[:, None] if vals.ndim == 1 else vals
    return np.concatenate([np.bincount(cell, vals[:, j], g ** 3) / cnt
                           for j in range(vals.shape[1])]).astype(np.float32)


# ----------------------------------------------------------------- probe
def ridge_path(X, Y, alphas):
    """Ridge weights for several alphas from one eigendecomposition. Uses the
    DUAL form (n x n) when there are fewer samples than descriptor dims -- the
    1840-dim ALL set made the primal d x d eigh the bottleneck of the run."""
    n, d = X.shape
    if n < d:
        lam, U = np.linalg.eigh(X @ X.T)
        UtY = U.T @ Y
        return {a: X.T @ (U @ (UtY / (lam + a)[:, None])) for a in alphas}
    lam, U = np.linalg.eigh(X.T @ X)
    UtXtY = U.T @ (X.T @ Y)
    return {a: U @ (UtXtY / (lam + a)[:, None]) for a in alphas}


def fit_predict(Xtr, Ytr, gtr, Xte, alphas=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-8] = 1.0
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ug = np.unique(gtr); np.random.default_rng(0).shuffle(ug)
    score = {a: 0.0 for a in alphas}
    for part in np.array_split(ug, 3):
        va = np.isin(gtr, part)
        W = ridge_path(Xtr[~va], Ytr[~va] - Ytr[~va].mean(0), alphas)
        for a in alphas:
            score[a] += np.linalg.norm(Xtr[va] @ W[a] + Ytr[~va].mean(0) - Ytr[va], axis=1).mean()
    best = min(score, key=score.get)
    W = ridge_path(Xtr, Ytr - Ytr.mean(0), (best,))[best]
    return Xte @ W + Ytr.mean(0)


def tangent_of(L, gi):
    for spec in CONTOUR_SPECS.values():
        s, e = spec["range"]
        if s <= gi < e:
            lo, hi = max(gi - 1, s), min(gi + 1, e - 1)
    t = L[hi] - L[lo]
    return t / np.linalg.norm(t)


def honest_relational(saved):
    """R5(b): ridge from the predicted anchor configuration to each anchor's true
    correction, cross-fitted over the SAME outer folds that produced the saved
    predictions, then scored end to end after re-interpolation."""
    d = np.load(ROOT / "results" / saved, allow_pickle=True)
    pred, truth = d["pred"], d["truth"]
    subj = np.array([str(s) for s in d["subject_id"]])
    folds = list(k_fold_subject_split(ds_ids_all[:200], 3, 0))
    fold_of = np.zeros(len(pred), int)
    for f, (_, te) in enumerate(folds):
        fold_of[np.isin(subj, list(te))] = f
    new = pred.copy()
    for gi in AI:
        X = (pred[:, AI] - pred[:, gi][:, None, :]).reshape(len(pred), -1)
        Y = truth[:, gi] - pred[:, gi]
        for f in range(3):
            tr, te = fold_of != f, fold_of == f
            new[te, gi] = pred[te, gi] + fit_predict(X[tr], Y[tr], subj[tr], X[te])
    out = pred.copy()
    for spec in CONTOUR_SPECS.values():
        a = spec["anchors"]
        for u, v in zip(a[:-1], a[1:]):
            su, sv = new[:, u] - pred[:, u], new[:, v] - pred[:, v]
            w = np.linspace(0, 1, v - u + 1)[None, :, None]
            out[:, u:v + 1] = pred[:, u:v + 1] + (1 - w) * su[:, None, :] + w * sv[:, None, :]
            for i in range(len(out)):
                out[i, u:v + 1] = resample_uniform(out[i, u:v + 1])
    out[:, AI] = new[:, AI]
    e0 = np.linalg.norm(pred - truth, axis=2); e1 = np.linalg.norm(out - truth, axis=2)
    dd = e1.mean(1) - e0.mean(1)
    print("\nR5(b) HONEST relational correction on real saved predictions (cross-fitted):")
    print(f"   overall {e0.mean():.4f} -> {e1.mean():.4f}  ({e1.mean()-e0.mean():+.4f})   "
          f"paired t = {dd.mean()/(dd.std(ddof=1)/np.sqrt(len(dd))):+.2f}")
    print(f"   anchors {e0[:, AI].mean():.4f} -> {e1[:, AI].mean():.4f}")
    for gi in AI:
        print(f"     anchor {gi:2d} ({ANCHOR_KIND[gi]:>11s})  {e0[:, gi].mean():.3f} -> {e1[:, gi].mean():.3f}")


ds_ids_all: list = []


def main():
    global ds_ids_all
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-subjects", type=int, default=120)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--crop", type=float, default=45.0)
    ap.add_argument("--saved", default="twopass_surfsnap.npz")
    ap.add_argument("--out", default="results/feature_probe_r2.npz")
    ap.add_argument("--sets", nargs="+", default=None,
                    help="only these feature sets (B_base is always included); fields "
                         "no chosen set needs are not computed")
    args = ap.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                 landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    ds_ids_all = list(ds.subject_ids)

    # honest relational test first: no meshes, answers R5 on its own
    honest_relational(args.saved)

    sv = np.load(ROOT / "results" / args.saved, allow_pickle=True)
    saved_pred = {f"{s}_{side}": p for s, side, p in
                  zip(map(str, sv["subject_id"]), map(str, sv["side"]), sv["pred"])}

    sets = list(FEATURE_SETS) if not args.sets else ["B_base"] + [x for x in args.sets if x != "B_base"]
    needed = {b for nm in sets for b in FEATURE_SETS[nm]}
    print(f"feature sets: {sets}", flush=True)
    rng = np.random.default_rng(0)       # same stream as the full run -> identical centres
    subjects = list(ds.subject_ids[:args.n_subjects])
    blocks: dict[str, list] = {}
    Y, GRP, ANC, TAN = [], [], [], []
    t0 = time.time()
    ears = list(iter_landmarks_only(ds, subjects))
    for ei, e in enumerate(ears):
        L = np.asarray(e.landmarks, float)
        key = f"{e.subject_id}_{e.side}"
        if key not in saved_pred:
            continue
        P_hat = saved_pred[key]
        try:
            mesh = load_canonical_mesh(ds, e.subject_id, e.side)
        except Exception as exc:
            print(f"  skip {key}: {exc}", flush=True); continue
        v_all = np.asarray(mesh.vertices)
        head_c = v_all.mean(0)
        keep = np.linalg.norm(v_all - L.mean(0), axis=1) <= args.crop
        fmask = keep[np.asarray(mesh.faces)].all(axis=1)
        sub = mesh.submesh([np.flatnonzero(fmask)], append=True)
        del mesh, v_all; gc.collect()
        V = np.asarray(sub.vertices, float); N = np.asarray(sub.vertex_normals, float)
        tree = cKDTree(V)

        # displaced centres for this ear, then only the vertices they need
        centres = []
        for gi in AI:
            for _ in range(args.k):
                off = rng.normal(scale=args.sigma, size=3)
                centres.append((gi, off, L[gi] + off))
        need = np.unique(np.concatenate(
            [np.asarray(tree.query_ball_point(c, R), dtype=int) for _, _, c in centres]))

        # per-vertex fields on `need` -- only those a chosen set uses
        C1 = _local_curvature(sub, np.arange(len(V)))[need]
        C1 = C1 / (np.abs(C1).mean() + 1e-12)
        if needed & {"si2", "cv2"}:
            k1a, k2a, _ = principal_curvatures(V, N, tree, need, k=48)
            si2, cv2 = shape_index(k1a, k2a), np.sqrt((k1a ** 2 + k2a ** 2) / 2)
            cv2 /= cv2.mean() + 1e-12
        if needed & {"si3", "cv3", "ori", "turn", "lsp"}:
            k1b, k2b, dirb = principal_curvatures(V, N, tree, need, k=120)
            si3, cv3 = shape_index(k1b, k2b), np.sqrt((k1b ** 2 + k2b ** 2) / 2)
            cv3 /= cv3.mean() + 1e-12
            ori = np.stack([dirb[:, 0] ** 2, dirb[:, 1] ** 2, dirb[:, 2] ** 2,
                            dirb[:, 0] * dirb[:, 1], dirb[:, 0] * dirb[:, 2], dirb[:, 1] * dirb[:, 2]], 1)
            ntree = cKDTree(V[need])
            _, nbn = ntree.query(V[need], k=min(32, len(need)))
            T = np.einsum("nki,nkj->nij", dirb[nbn], dirb[nbn]) / nbn.shape[1]
            turn = 1.0 - np.linalg.eigvalsh(T)[:, -1]
        if "encl" in needed:
            dd, nb6 = tree.query(V[need], k=160, distance_upper_bound=6.0)
            valid = np.isfinite(dd)
            nb6c = np.where(valid, nb6, 0)
            hgt = ((V[nb6c] - V[need][:, None, :]) * N[need][:, None, :]).sum(2)
            encl = ((hgt > 1.0) & valid).sum(1) / np.maximum(valid.sum(1), 1)
        if "depth" in needed:
            cc = V.mean(0); pn = np.linalg.svd(V - cc, full_matrices=False)[2][2]
            if np.dot(pn, cc - head_c) < 0:
                pn = -pn
            depth = (V[need] - cc) @ pn
        if needed & {"hks", "wks"}:
            hks, wks = graph_hks_wks(V)
            hks, wks = hks[need], wks[need]

        for gi, off, c in centres:
            ids = np.asarray(tree.query_ball_point(c, R), dtype=int)
            li = np.searchsorted(need, ids)          # need is sorted (np.unique)
            rel = (V[ids] - c) / R
            b = {}
            if len(ids) < 10:
                continue
            cell3 = np.clip(((rel + 1.0) * 0.5 * G).astype(int), 0, G - 1)
            cell = cell3[:, 0] * G * G + cell3[:, 1] * G + cell3[:, 2]
            b["occ"] = (np.bincount(cell, minlength=G ** 3) / len(ids)).astype(np.float32)
            b["nrm"] = grid_means(rel, N[ids], G)
            b["curv"] = grid_means(rel, C1[li], G)
            if needed & {"si2", "cv2"}:
                b["si2"] = grid_means(rel, si2[li], G); b["cv2"] = grid_means(rel, cv2[li], G)
            if needed & {"si3", "cv3", "ori", "turn", "lsp"}:
                b["si3"] = grid_means(rel, si3[li], G); b["cv3"] = grid_means(rel, cv3[li], G)
                b["ori"] = grid_means(rel, ori[li], G); b["turn"] = grid_means(rel, turn[li], G)
            if "depth" in needed:
                b["depth"] = grid_means(rel, depth[li], G)
            if "encl" in needed:
                b["encl"] = grid_means(rel, encl[li], G)
            if needed & {"hks", "wks"}:
                b["hks"] = grid_means(rel, hks[li], GH); b["wks"] = grid_means(rel, wks[li], GH)
            nc = N[ids[np.argmin(np.linalg.norm(V[ids] - c, axis=1))]]
            if "spin" in needed:
                p = V[ids] - c; beta = p @ nc
                alpha = np.sqrt(np.maximum((p ** 2).sum(1) - beta ** 2, 0))
                b["spin"] = (np.histogram2d(alpha, beta, bins=10, range=[[0, R], [-4, 4]])[0]
                             .ravel() / len(ids)).astype(np.float32)
            if "lsp" in needed:
                b["lsp"] = (np.histogram2d(si3[li], N[ids] @ nc, bins=10, range=[[-1, 1], [-1, 1]])[0]
                            .ravel() / len(ids)).astype(np.float32)
            if "rel" in needed:
                others = [j for j in AI if j != gi]
                vec = (c - P_hat[others])
                b["rel"] = np.concatenate([vec.ravel(), np.linalg.norm(vec, axis=1)]).astype(np.float32)
            for kname, arr in b.items():
                blocks.setdefault(kname, []).append(arr)
            Y.append(-off); GRP.append(e.subject_id); ANC.append(gi); TAN.append(tangent_of(L, gi))
        del sub, V, N, tree; gc.collect()
        if (ei + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {ei+1}/{len(ears)} ears  {el/60:.1f} min  "
                  f"(~{el/(ei+1)*(len(ears)-ei-1)/60:.0f} min left)", flush=True)

    Y = np.array(Y); GRP = np.array(GRP); ANC = np.array(ANC); TAN = np.array(TAN)
    blocks = {k: np.array(v, np.float32) for k, v in blocks.items()}
    for k, v in blocks.items():
        bad = ~np.isfinite(v).all(1)
        if bad.any():
            print(f"  WARNING block {k}: {bad.sum()} of {len(v)} rows non-finite -> zeroed", flush=True)
            v[bad] = 0.0
    print(f"\nextracted {len(Y)} displaced patches in {(time.time()-t0)/60:.1f} min", flush=True)

    usubj = np.array(sorted(set(GRP))); np.random.default_rng(1).shuffle(usubj)
    fold_of = np.zeros(len(Y), int)
    for f, fs in enumerate(np.array_split(usubj, 3)):
        fold_of[np.isin(GRP, fs)] = f

    names = sets
    ALONG = {nm: np.zeros(len(Y)) for nm in names}
    ERR = {nm: np.zeros(len(Y)) for nm in names}
    for nm in names:
        X = np.concatenate([blocks[b] for b in FEATURE_SETS[nm]], axis=1)
        for gi in AI:
            m = ANC == gi
            for f in range(3):
                tr, te = m & (fold_of != f), m & (fold_of == f)
                r = fit_predict(X[tr], Y[tr], GRP[tr], X[te]) - Y[te]
                ALONG[nm][te] = np.abs((r * TAN[te]).sum(1))
                ERR[nm][te] = np.linalg.norm(r, axis=1)
        print(f"  probed {nm} ({X.shape[1]} dims)", flush=True)

    none_al = np.abs((Y * TAN).sum(1))

    def per_subj(v, m):
        u, inv = np.unique(GRP[m], return_inverse=True)
        return np.bincount(inv, v[m]) / np.bincount(inv)

    print("\n" + "=" * 110)
    print("ALONG-CONTOUR residual (mm) per anchor -- lower = the input locates the point along the contour")
    print(f"{'anchor':>6s} {'kind':>11s} {'none':>6s} " + " ".join(f"{n[:10]:>10s}" for n in names))
    for gi in AI:
        m = ANC == gi
        print(f"{gi:6d} {ANCHOR_KIND[gi]:>11s} {none_al[m].mean():6.3f} "
              + " ".join(f"{ALONG[n][m].mean():10.3f}" for n in names))

    print("\nvs B_base on the ALONG axis, paired over subjects, |z| > 3 only:")
    for gi in AI:
        m = ANC == gi
        base = per_subj(ALONG["B_base"], m)
        for nm in names[1:]:
            d = per_subj(ALONG[nm], m) - base
            z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
            if abs(z) > 3:
                print(f"  anchor {gi:2d} ({ANCHOR_KIND[gi]:>11s}): {nm:14s} "
                      f"{'BETTER' if d.mean() < 0 else 'worse '} by {abs(d.mean()):.3f} mm  z={z:+.1f}")

    print("\npooled over all anchors:")
    for nm in names:
        print(f"  {nm:14s} along {ALONG[nm].mean():.3f}   total {ERR[nm].mean():.3f}")

    chosen = np.zeros(len(Y)); picks = {}
    for gi in AI:
        for f in range(3):
            trm = (ANC == gi) & (fold_of != f); tem = (ANC == gi) & (fold_of == f)
            best = min(names, key=lambda n: ALONG[n][trm].mean())
            picks.setdefault(gi, []).append(best); chosen[tem] = ALONG[best][tem]
    print(f"\nper-anchor choice, CROSS-FITTED: along {chosen.mean():.3f}  vs B_base {ALONG['B_base'].mean():.3f}")
    for gi in AI:
        print(f"  anchor {gi:2d} ({ANCHOR_KIND[gi]:>11s}) picks: {picks[gi]}")

    np.savez_compressed(ROOT / args.out, anchor=ANC, group=GRP, fold=fold_of, y=Y, tangent=TAN,
                        **{f"along_{n}": ALONG[n] for n in names},
                        **{f"err_{n}": ERR[n] for n in names})
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
