"""Local evidence x spatial configuration, SCN-style, by Hough voting.

SpatialConfiguration-Net (Payer, Stern, Bischof & Urschler, Medical Image
Analysis 2019) splits landmark localisation into "locally accurate but
ambiguous candidate predictions" and a spatial-configuration component that
resolves the ambiguity, and outperforms related methods on size-limited
datasets. That is exactly the pair of information sources this project has
found: LOCAL surface evidence (usable for junction anchors) and RELATIONAL
evidence (T2, usable for the sliders). So far they were applied in sequence;
SCN combines them multiplicatively.

Post-hoc implementation with no new network:
  * local regressor per anchor (ridge on the round-2 descriptor families,
    trained on synthetic displacements + real out-of-fold positions of
    training ears -- the same model as local_refine.py)
  * at test time, ~30 surface candidates within 4mm of the current best
    position; each casts a VOTE = candidate + predicted displacement
    (regression voting, as in random-forest landmark voting)
  * local likelihood = kernel density of the votes; spatial prior = isotropic
    Gaussian centred on the current best position (which already contains the
    relational T2 correction), width = that anchor's RMS error on training folds
  * estimate = prior-weighted mean-shift mode of the votes
  * per-anchor blend step toward it chosen on inner folds; fully nested
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

import local_refine as LR
from feature_probe_r2 import fit_predict, grid_means, principal_curvatures, shape_index
from feature_probe_r3 import AI, KIND, STEPS, load_sorted, rebuild
from src.correction.patch_features import _local_curvature
from src.foundations.canonical import load_canonical_mesh
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split

CAND_CACHE = ROOT / "results" / "scn_candidates.npz"
N_CAND, CAND_R = 30, 4.0


def candidate_descriptors(best, subj, side, keys):
    if CAND_CACHE.exists():
        c = np.load(CAND_CACHE, allow_pickle=True)
        if list(c["keys"]) == list(keys):
            return {k: c[k] for k in c.files}
    ds = Dataset(mesh_dir=str(LR.DATA / "mesh"), landmarks_dir=str(LR.DATA / "landmarks"))
    blocks = {b: [] for b in LR.LOCAL}; meta = {"ear": [], "anchor": [], "pos": []}
    t0 = time.time()
    for i in range(len(keys)):
        m = load_canonical_mesh(ds, subj[i], side[i])
        va = np.asarray(m.vertices); head_c = va.mean(0)
        keep = np.linalg.norm(va - best[i].mean(0), axis=1) <= 45.0
        sub = m.submesh([np.flatnonzero(keep[np.asarray(m.faces)].all(1))], append=True)
        del m, va; gc.collect()
        V = np.asarray(sub.vertices, float); N = np.asarray(sub.vertex_normals, float); tree = cKDTree(V)
        centres = []
        for gi in AI:
            ball = np.asarray(tree.query_ball_point(best[i, gi], CAND_R), dtype=int)
            if len(ball) < N_CAND:
                ball = np.asarray(tree.query(best[i, gi], k=N_CAND)[1], dtype=int)
            # farthest-point subsample for even coverage of the search disc
            sel = [int(np.argmin(np.linalg.norm(V[ball] - best[i, gi], axis=1)))]
            d = np.linalg.norm(V[ball] - V[ball[sel[0]]], axis=1)
            while len(sel) < N_CAND:
                j = int(np.argmax(d)); sel.append(j); d = np.minimum(d, np.linalg.norm(V[ball] - V[ball[j]], axis=1))
            for j in sel:
                centres.append((gi, V[ball[j]]))
        need = np.unique(np.concatenate([np.asarray(tree.query_ball_point(c, LR.R), dtype=int) for _, c in centres]))
        C1 = _local_curvature(sub, np.arange(len(V)))[need]; C1 /= np.abs(C1).mean() + 1e-12
        k1a, k2a, _ = principal_curvatures(V, N, tree, need, k=48)
        k1b, k2b, dirb = principal_curvatures(V, N, tree, need, k=120)
        si2, cv2 = shape_index(k1a, k2a), np.sqrt((k1a ** 2 + k2a ** 2) / 2); cv2 /= cv2.mean() + 1e-12
        si3, cv3 = shape_index(k1b, k2b), np.sqrt((k1b ** 2 + k2b ** 2) / 2); cv3 /= cv3.mean() + 1e-12
        ori = np.stack([dirb[:, 0] ** 2, dirb[:, 1] ** 2, dirb[:, 2] ** 2, dirb[:, 0] * dirb[:, 1],
                        dirb[:, 0] * dirb[:, 2], dirb[:, 1] * dirb[:, 2]], 1)
        _, nbn = cKDTree(V[need]).query(V[need], k=min(32, len(need)))
        turn = 1 - np.linalg.eigvalsh(np.einsum("nki,nkj->nij", dirb[nbn], dirb[nbn]) / nbn.shape[1])[:, -1]
        dd, nb6 = tree.query(V[need], k=160, distance_upper_bound=6.0)
        valid = np.isfinite(dd); nb6c = np.where(valid, nb6, 0)
        hgt = ((V[nb6c] - V[need][:, None, :]) * N[need][:, None, :]).sum(2)
        encl = ((hgt > 1.0) & valid).sum(1) / np.maximum(valid.sum(1), 1)
        cc = V.mean(0); pn = np.linalg.svd(V - cc, full_matrices=False)[2][2]
        if np.dot(pn, cc - head_c) < 0:
            pn = -pn
        depth = (V[need] - cc) @ pn
        for gi, c in centres:
            ids = np.asarray(tree.query_ball_point(c, LR.R), dtype=int)
            li = np.clip(np.searchsorted(need, ids), 0, len(need) - 1)
            rel = (V[ids] - c) / LR.R
            cell3 = np.clip(((rel + 1.0) * 0.5 * LR.G).astype(int), 0, LR.G - 1)
            cell = cell3[:, 0] * LR.G ** 2 + cell3[:, 1] * LR.G + cell3[:, 2]
            nc = N[ids[np.argmin(np.linalg.norm(V[ids] - c, axis=1))]]
            p = V[ids] - c; beta = p @ nc; alpha = np.sqrt(np.maximum((p ** 2).sum(1) - beta ** 2, 0))
            b = {"occ": (np.bincount(cell, minlength=LR.G ** 3) / len(ids)).astype(np.float32),
                 "nrm": grid_means(rel, N[ids], LR.G), "curv": grid_means(rel, C1[li], LR.G),
                 "si2": grid_means(rel, si2[li], LR.G), "cv2": grid_means(rel, cv2[li], LR.G),
                 "si3": grid_means(rel, si3[li], LR.G), "cv3": grid_means(rel, cv3[li], LR.G),
                 "ori": grid_means(rel, ori[li], LR.G), "turn": grid_means(rel, turn[li], LR.G),
                 "depth": grid_means(rel, depth[li], LR.G), "encl": grid_means(rel, encl[li], LR.G),
                 "spin": (np.histogram2d(alpha, beta, bins=10, range=[[0, LR.R], [-4, 4]])[0].ravel() / len(ids)).astype(np.float32),
                 "lsp": (np.histogram2d(si3[li], N[ids] @ nc, bins=10, range=[[-1, 1], [-1, 1]])[0].ravel() / len(ids)).astype(np.float32)}
            for kk in LR.LOCAL:
                blocks[kk].append(b[kk])
            meta["ear"].append(i); meta["anchor"].append(gi); meta["pos"].append(c)
        del sub, V, N, tree; gc.collect()
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  candidates {i+1}/{len(keys)}  {el/60:.1f} min  (~{el/(i+1)*(len(keys)-i-1)/60:.0f} min left)", flush=True)
    out = {k: np.array(v, np.float32) for k, v in blocks.items()}
    out.update({k: np.array(v) for k, v in meta.items()}); out["keys"] = keys
    np.savez_compressed(CAND_CACHE, **out)
    return out


def mode_of_votes(votes, prior_c, sig_p, bw=0.75, iters=5):
    """Prior-weighted mean-shift on the vote cloud."""
    x = votes.mean(0)
    wp = np.exp(-np.sum((votes - prior_c) ** 2, 1) / (2 * sig_p ** 2))
    for _ in range(iters):
        wk = np.exp(-np.sum((votes - x) ** 2, 1) / (2 * bw ** 2)) * wp
        if wk.sum() < 1e-12:
            break
        x = (wk[:, None] * votes).sum(0) / wk.sum()
    return x


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    best = np.load(ROOT / "results" / "t2_on_average.npy")
    n = len(keys)
    ds = Dataset(mesh_dir=str(LR.DATA / "mesh"), landmarks_dir=str(LR.DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f
    D = LR.extract(best, truth, subj, side, keys)                 # training samples (cached)
    Cd = candidate_descriptors(best, subj, side, keys)
    X_tr_all = np.hstack([D[b] for b in LR.LOCAL]); X_c_all = np.hstack([Cd[b] for b in LR.LOCAL])
    base_err = np.linalg.norm(best - truth, axis=2)
    print(f"\ncurrent best {base_err.mean():.4f}")

    variants = {"local votes only": False, "local votes x spatial prior (SCN)": True}
    out = {v: best.copy() for v in variants}
    for gi in AI:
        mtr = D["anchor"] == gi; mc = Cd["anchor"] == gi
        for f in range(3):
            tr_e = np.flatnonzero(fold_of != f); te_e = np.flatnonzero(fold_of == f)
            sig_p = max(np.sqrt((base_err[tr_e, gi] ** 2).mean()), 0.3)
            us = np.array(sorted(set(subj[tr_e]))); np.random.default_rng(f).shuffle(us)
            inner = [tr_e[np.isin(subj[tr_e], p)] for p in np.array_split(us, 2)]

            def estimates(train_ears, eval_ears, use_prior):
                trm = mtr & np.isin(D["ear"], train_ears); cm = mc & np.isin(Cd["ear"], eval_ears)
                disp = fit_predict(X_tr_all[trm], D["target"][trm], subj[D["ear"][trm]], X_c_all[cm])
                votes = Cd["pos"][cm] + disp; ears = Cd["ear"][cm]
                est = {}
                for e_i in np.unique(ears):
                    v = votes[ears == e_i]
                    est[e_i] = mode_of_votes(v, best[e_i, gi], sig_p if use_prior else 1e6)
                return est

            for vname, use_prior in variants.items():
                num = np.zeros(len(STEPS))
                for j in range(2):
                    est = estimates(inner[1 - j], inner[j], use_prior)
                    for e_i, x in est.items():
                        for si, s_ in enumerate(STEPS):
                            num[si] += np.linalg.norm(best[e_i, gi] + s_ * (x - best[e_i, gi]) - truth[e_i, gi])
                w = STEPS[int(np.argmin(num))]
                for e_i, x in estimates(tr_e, te_e, use_prior).items():
                    out[vname][e_i, gi] = best[e_i, gi] + w * (x - best[e_i, gi])
        print(f"  anchor {gi:2d} ({KIND[gi]:>11s}) done", flush=True)

    for vname in variants:
        e = np.linalg.norm(rebuild(best, out[vname]) - truth, axis=2); d = e.mean(1) - base_err.mean(1)
        print(f"\n{vname:36s} {e.mean():.4f}  vs best {d.mean():+.4f} (t {d.mean()/(d.std(ddof=1)/np.sqrt(n)+1e-12):+.2f})")
        print("   per anchor: " + "  ".join(f"{g}:{base_err[:, g].mean():.2f}->{e[:, g].mean():.2f}" for g in AI))
    np.savez_compressed(ROOT / "results" / "scn_voting.npz", keys=keys, votes_only=out["local votes only"],
                        scn=out["local votes x spatial prior (SCN)"])


if __name__ == "__main__":
    main()
