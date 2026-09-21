"""Local descriptor refinement of the current best predictions.

Strategic basis: if the 11 non-slider anchors were perfect (sliders untouched),
the overall error would be 0.9663mm -- 1.0 does not depend on the sliders. A
-40% cut on those 11 plus -20% on the sliders reaches 0.993. Those 11 are the
anchors where every probe found real LOCAL information (junctions recover
55-60% of a synthetic displacement), and where the literature descriptors gave
small significant probe wins (ridge direction: 46, 54, 33; shape index: 0, 24).
Free-click anchors that carry individual information (not predictable from the
other true anchors): 0, 24, 42, 84, 25.

Question: does that local information translate into a REAL error reduction on
our actual best predictions (average of two runs + T2, 1.2905mm)?

METHOD
  training samples, per ear and anchor: 8 synthetic displacements (sigma 1.5mm)
    around the TRUE landmark, plus the real best-prediction position (a realistic
    error example); target = true position - sample centre
  descriptors (7mm patch, 4^3 grid): occupancy, normals, one-ring curvature
    [BASE]; + shape index & curvedness at 2 and 3mm, ridge-direction tensor and
    turning, spin image, Local Surface Patch histogram, depth, fold enclosure
    [LOCAL]
  per-anchor ridge, fully nested: inner folds train on displaced + real samples
    and pick the step on held-out REAL samples; the outer model corrects the
    test fold's real positions once
  crops centred on the PREDICTED landmark centroid (no ground truth at test)
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

from feature_probe_r2 import fit_predict, grid_means, principal_curvatures, shape_index
from feature_probe_r3 import AI, KIND, STEPS, load_sorted, rebuild
from src.correction.patch_features import _local_curvature
from src.foundations.canonical import load_canonical_mesh
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split

DATA = ROOT / "2026 Munich Tech Arena - Datas"
CACHE = ROOT / "results" / "local_refine_desc.npz"
G, R, KDISP, SIG = 4, 7.0, 8, 1.5
BASE = ["occ", "nrm", "curv"]
LOCAL = BASE + ["si2", "cv2", "si3", "cv3", "ori", "turn", "spin", "lsp", "depth", "encl"]
SLIDERS = [6, 22, 64, 74]


def extract(best, truth, subj, side, keys):
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        if list(c["keys"]) == list(keys):
            return {k: c[k] for k in c.files}
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    rng = np.random.default_rng(0)
    blocks = {b: [] for b in LOCAL}
    meta = {"ear": [], "anchor": [], "real": [], "target": []}
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
            for _ in range(KDISP):
                centres.append((gi, 0, truth[i, gi] + rng.normal(scale=SIG, size=3)))
            centres.append((gi, 1, best[i, gi]))
        need = np.unique(np.concatenate([np.asarray(tree.query_ball_point(c, R), dtype=int) for _, _, c in centres]))
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
        for gi, real, c in centres:
            ids = np.asarray(tree.query_ball_point(c, R), dtype=int)
            if len(ids) < 10:
                ids = np.asarray(tree.query(c, k=10)[1], dtype=int)
            li = np.searchsorted(need, ids); li = np.clip(li, 0, len(need) - 1)
            rel = (V[ids] - c) / R
            cell3 = np.clip(((rel + 1.0) * 0.5 * G).astype(int), 0, G - 1)
            cell = cell3[:, 0] * G * G + cell3[:, 1] * G + cell3[:, 2]
            nc = N[ids[np.argmin(np.linalg.norm(V[ids] - c, axis=1))]]
            p = V[ids] - c; beta = p @ nc; alpha = np.sqrt(np.maximum((p ** 2).sum(1) - beta ** 2, 0))
            b = {"occ": (np.bincount(cell, minlength=G ** 3) / len(ids)).astype(np.float32),
                 "nrm": grid_means(rel, N[ids], G), "curv": grid_means(rel, C1[li], G),
                 "si2": grid_means(rel, si2[li], G), "cv2": grid_means(rel, cv2[li], G),
                 "si3": grid_means(rel, si3[li], G), "cv3": grid_means(rel, cv3[li], G),
                 "ori": grid_means(rel, ori[li], G), "turn": grid_means(rel, turn[li], G),
                 "depth": grid_means(rel, depth[li], G), "encl": grid_means(rel, encl[li], G),
                 "spin": (np.histogram2d(alpha, beta, bins=10, range=[[0, R], [-4, 4]])[0].ravel() / len(ids)).astype(np.float32),
                 "lsp": (np.histogram2d(si3[li], N[ids] @ nc, bins=10, range=[[-1, 1], [-1, 1]])[0].ravel() / len(ids)).astype(np.float32)}
            for kk in LOCAL:
                blocks[kk].append(b[kk])
            meta["ear"].append(i); meta["anchor"].append(gi); meta["real"].append(real)
            meta["target"].append(truth[i, gi] - c)
        del sub, V, N, tree; gc.collect()
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(keys)} ears  {el/60:.1f} min  (~{el/(i+1)*(len(keys)-i-1)/60:.0f} min left)", flush=True)
    out = {k: np.array(v, np.float32) for k, v in blocks.items()}
    out.update({k: np.array(v) for k, v in meta.items()}); out["keys"] = keys
    np.savez_compressed(CACHE, **out)
    return out


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    best = np.load(ROOT / "results" / "t2_on_average.npy")
    n = len(keys)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f

    print("extracting descriptors (cached after the first run)...", flush=True)
    D = extract(best, truth, subj, side, keys)
    ear, anc, real, tgt = D["ear"], D["anchor"], D["real"].astype(bool), D["target"]
    base_err = np.linalg.norm(best - truth, axis=2)
    print(f"\ncurrent best {base_err.mean():.4f}")

    results = {}
    for setname, names in (("BASE", BASE), ("LOCAL", LOCAL)):
        X = np.hstack([D[b] for b in names])
        corrected = best.copy()
        steps = {}
        for gi in AI:
            m = anc == gi
            for f in range(3):
                tr_ears = np.flatnonzero(fold_of != f); te_ears = np.flatnonzero(fold_of == f)
                us = np.array(sorted(set(subj[tr_ears]))); np.random.default_rng(f).shuffle(us)
                inner = [tr_ears[np.isin(subj[tr_ears], p)] for p in np.array_split(us, 2)]
                num = np.zeros(len(STEPS))
                for j in range(2):
                    a_e, b_e = inner[1 - j], inner[j]
                    trm = m & np.isin(ear, a_e); vam = m & np.isin(ear, b_e) & real
                    pr = fit_predict(X[trm], tgt[trm], subj[ear[trm]], X[vam])
                    for si, s_ in enumerate(STEPS):
                        num[si] += np.linalg.norm(s_ * pr - tgt[vam], axis=1).sum()
                w = STEPS[int(np.argmin(num))]
                steps.setdefault(gi, []).append(w)
                trm = m & np.isin(ear, tr_ears); tem = m & np.isin(ear, te_ears) & real
                pr = fit_predict(X[trm], tgt[trm], subj[ear[trm]], X[tem])
                for e_i, corr in zip(ear[tem], pr):
                    corrected[e_i, gi] = best[e_i, gi] + w * corr
            print(f"  [{setname}] anchor {gi:2d} ({KIND[gi]:>11s}) steps {steps[gi]}", flush=True)
        results[setname] = (corrected, steps)

    def score(label, anchors_from, which):
        na = best.copy()
        for g in which:
            na[:, g] = anchors_from[:, g]
        e = np.linalg.norm(rebuild(best, na) - truth, axis=2); d = e.mean(1) - base_err.mean(1)
        print(f"  {label:36s} {e.mean():.4f}  vs best {d.mean():+.4f} (t {d.mean()/(d.std(ddof=1)/np.sqrt(n)+1e-12):+.2f})")
        return e

    print("\nend to end, starting from the current best (1.2905):")
    nonsl = [g for g in AI if g not in SLIDERS]
    for setname in ("BASE", "LOCAL"):
        corrected = results[setname][0]
        score(f"{setname} refine, 11 non-slider anchors", corrected, nonsl)
        e = score(f"{setname} refine, all 15 anchors", corrected, AI)
        print("     per anchor: " + "  ".join(f"{g}:{base_err[:, g].mean():.2f}->{e[:, g].mean():.2f}" for g in AI))
    np.savez_compressed(ROOT / "results" / "local_refine.npz", keys=keys,
                        base=results["BASE"][0], local=results["LOCAL"][0])


if __name__ == "__main__":
    main()
