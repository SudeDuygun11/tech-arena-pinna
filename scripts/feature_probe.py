"""Which input features carry position information for WHICH landmark?

Measured BEFORE any network training, because a feature-ablation training run
without prior evidence of signal is likely to come back null -- the lesson of
the two-pass run, which cost a night and returned t = -0.87.

METHOD (a linear probe, the standard cheap test of information content)
----------------------------------------------------------------------
For every anchor on every ear, draw K random displacements of the patch centre
(isotropic Gaussian, sigma = 1.5mm, the same as the training jitter). Extract
the local surface around each displaced centre and summarise it with a spatial
descriptor: a 5x5x5 grid over the patch, recording per cell the occupancy (pure
geometry) and the mean of each feature channel. A ridge regression per anchor
and per feature set then predicts the displacement back from the descriptor.

  * If a feature set lets the probe recover the true position, the information
    is present in that input for that landmark.
  * Reported split ALONG vs ACROSS the contour, since along-contour sliding is
    68% of anchor error and the only axis that can reach 1.0mm.
  * All feature sets see IDENTICAL displaced centres (paired comparison).

HONEST LIMITS
-------------
  * A linear probe under-reads what a deep network can extract. A probe WIN is
    strong evidence of signal; a probe NULL is not proof of absence.
  * Curvature here is computed once per vertex on the cropped mesh, not
    re-computed inside each patch as extract_patch does -- slightly different
    at patch borders; fine for measuring information, not a replica.

LEAKAGE AND SELECTION BIAS
--------------------------
3 outer folds over subjects (both ears together). Ridge alpha chosen by inner
grouped CV on training subjects only. Per-landmark "best feature set" is chosen
CROSS-FITTED (on the other two folds' test errors) -- choosing on the same
errors you report inflated a gain ~4x in scripts/per_part_posthoc.py.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.correction.patch_features import _gaussian_curvature, _local_curvature
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS
from src.foundations.dataset import Dataset

G = 5
RADII = (7.0, 11.0, 14.0)
FEATURE_SETS = {
    "G_geometry":    [("occ", 7.0)],
    "F0_base":       [("occ", 7.0), ("nrm", 7.0), ("curv", 7.0)],
    "F1_no_curv":    [("occ", 7.0), ("nrm", 7.0)],
    "F2_no_normal":  [("occ", 7.0), ("curv", 7.0)],
    "F3_plus_gauss": [("occ", 7.0), ("nrm", 7.0), ("curv", 7.0), ("gauss", 7.0)],
    "F4_11mm":       [("occ", 11.0), ("nrm", 11.0), ("curv", 11.0)],
    "F5_multiscale": [("occ", 7.0), ("nrm", 7.0), ("curv", 7.0),
                      ("occ", 14.0), ("nrm", 14.0), ("curv", 14.0)],
}
ANCHOR_KIND = {0: "junction", 6: "extremum", 22: "extremum", 24: "junction",
               25: "junction", 33: "junction", 42: "junction", 46: "junction",
               50: "junction", 54: "junction", 55: "height-ref", 64: "cross-ref",
               74: "constructed", 75: "junction", 84: "junction"}


def describe(V, N, C, K, tree, centre, radius):
    """Spatial grid descriptor of the surface within `radius` of `centre`."""
    ids = tree.query_ball_point(centre, radius)
    out = {"occ": np.zeros(G**3, np.float32), "nrm": np.zeros(3 * G**3, np.float32),
           "curv": np.zeros(G**3, np.float32), "gauss": np.zeros(G**3, np.float32)}
    if len(ids) < 10:
        return out
    ids = np.asarray(ids)
    rel = (V[ids] - centre) / radius
    cell3 = np.clip(((rel + 1.0) * 0.5 * G).astype(int), 0, G - 1)
    cell = cell3[:, 0] * G * G + cell3[:, 1] * G + cell3[:, 2]
    cnt = np.bincount(cell, minlength=G**3).astype(np.float32)
    den = np.maximum(cnt, 1.0)
    out["occ"] = cnt / len(ids)
    out["nrm"] = np.concatenate([np.bincount(cell, N[ids, k], G**3) / den
                                 for k in range(3)]).astype(np.float32)
    out["curv"] = (np.bincount(cell, C[ids], G**3) / den).astype(np.float32)
    out["gauss"] = (np.bincount(cell, K[ids], G**3) / den).astype(np.float32)
    return out


def ridge_path(X, Y, alphas):
    """Ridge weights for several alphas from one eigendecomposition."""
    lam, U = np.linalg.eigh(X.T @ X)
    UtXtY = U.T @ (X.T @ Y)
    return {a: U @ (UtXtY / (lam + a)[:, None]) for a in alphas}


def fit_predict(Xtr, Ytr, gtr, Xte, alphas=(1.0, 10.0, 100.0, 1000.0)):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = Ytr.mean(0)
    # inner grouped CV (by subject) to choose alpha
    ug = np.unique(gtr); rng = np.random.default_rng(0); rng.shuffle(ug)
    parts = np.array_split(ug, 3)
    score = {a: 0.0 for a in alphas}
    for p in parts:
        va = np.isin(gtr, p)
        W = ridge_path(Xtr[~va], Ytr[~va] - Ytr[~va].mean(0), alphas)
        for a in alphas:
            r = Xtr[va] @ W[a] + Ytr[~va].mean(0) - Ytr[va]
            score[a] += np.linalg.norm(r, axis=1).mean()
    best = min(score, key=score.get)
    W = ridge_path(Xtr, Ytr - ym, (best,))[best]
    return Xte @ W + ym, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-subjects", type=int, default=90)
    ap.add_argument("--k", type=int, default=10, help="displacements per anchor per ear")
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--crop", type=float, default=50.0)
    ap.add_argument("--out", default="results/feature_probe.npz")
    args = ap.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                 landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    subjects = list(ds.subject_ids[:args.n_subjects])
    rng = np.random.default_rng(0)
    AI = list(ANCHOR_INDICES)

    # ------------------------------------------------ extraction (one pass)
    blocks = {(b, r): [] for b in ("occ", "nrm", "curv", "gauss") for r in RADII}
    Y, GRP, ANC, TAN = [], [], [], []
    t0 = time.time()
    ears = list(iter_landmarks_only(ds, subjects))
    for ei, e in enumerate(ears):
        L = np.asarray(e.landmarks, float)
        try:
            mesh = load_canonical_mesh(ds, e.subject_id, e.side)
        except Exception as exc:
            print(f"  skip {e.subject_id}_{e.side}: {exc}", flush=True)
            continue
        v_all = np.asarray(mesh.vertices)
        keep = np.linalg.norm(v_all - L.mean(0), axis=1) <= args.crop
        fmask = keep[np.asarray(mesh.faces)].all(axis=1)
        sub = mesh.submesh([np.flatnonzero(fmask)], append=True)
        del mesh, v_all
        gc.collect()
        V = np.asarray(sub.vertices, float)
        N = np.asarray(sub.vertex_normals, float)
        allv = np.arange(len(V))
        C = _local_curvature(sub, allv)
        K = _gaussian_curvature(sub, allv)
        C = C / (np.abs(C).mean() + 1e-12)          # per-ear scale, as render_dataset does
        K = K / (np.abs(K).mean() + 1e-12)
        tree = cKDTree(V)

        for gi in AI:
            for spec in CONTOUR_SPECS.values():
                s, en = spec["range"]
                if s <= gi < en:
                    lo, hi = max(gi - 1, s), min(gi + 1, en - 1)
            t = L[hi] - L[lo]; t /= np.linalg.norm(t)
            for _ in range(args.k):
                off = rng.normal(scale=args.sigma, size=3)
                c = L[gi] + off
                for r in RADII:
                    dsc = describe(V, N, C, K, tree, c, r)
                    for b in ("occ", "nrm", "curv", "gauss"):
                        blocks[(b, r)].append(dsc[b])
                Y.append(-off); GRP.append(e.subject_id); ANC.append(gi); TAN.append(t)
        del sub, V, N, C, K, tree
        gc.collect()
        if (ei + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {ei+1}/{len(ears)} ears  {el/60:.1f} min  "
                  f"(~{el/(ei+1)*(len(ears)-ei-1)/60:.0f} min left)", flush=True)

    Y = np.array(Y); GRP = np.array(GRP); ANC = np.array(ANC); TAN = np.array(TAN)
    blocks = {k: np.array(v, np.float32) for k, v in blocks.items()}
    print(f"\nextracted {len(Y)} displaced patches in {(time.time()-t0)/60:.1f} min\n", flush=True)

    # ------------------------------------------------ probe, 3 outer folds
    usubj = np.array(sorted(set(GRP))); rng2 = np.random.default_rng(1); rng2.shuffle(usubj)
    folds = np.array_split(usubj, 3)
    fold_of = np.zeros(len(Y), int)
    for f, fs in enumerate(folds):
        fold_of[np.isin(GRP, fs)] = f

    names = list(FEATURE_SETS)
    ERR = {nm: np.zeros(len(Y)) for nm in names}
    ALONG = {nm: np.zeros(len(Y)) for nm in names}
    ACROSS = {nm: np.zeros(len(Y)) for nm in names}
    for nm in names:
        X = np.concatenate([blocks[b] for b in FEATURE_SETS[nm]], axis=1)
        for gi in AI:
            m = ANC == gi
            for f in range(3):
                tr = m & (fold_of != f); te = m & (fold_of == f)
                pred, _ = fit_predict(X[tr], Y[tr], GRP[tr], X[te])
                r = pred - Y[te]                     # residual displacement
                a = (r * TAN[te]).sum(1)
                ERR[nm][te] = np.linalg.norm(r, axis=1)
                ALONG[nm][te] = np.abs(a)
                ACROSS[nm][te] = np.sqrt(np.maximum((r**2).sum(1) - a**2, 0))
        print(f"  probed {nm}", flush=True)

    none_err = np.linalg.norm(Y, axis=1)
    none_al = np.abs((Y * TAN).sum(1))

    # per-ear aggregation for paired statistics (samples within an ear correlate)
    ear_key = np.array([f"{g}" for g in GRP])

    def per_ear(v, m):
        keys = ear_key[m]; vals = v[m]
        u, inv = np.unique(keys, return_inverse=True)
        return np.bincount(inv, vals) / np.bincount(inv)

    print("\n" + "=" * 100)
    print("ALONG-CONTOUR residual (mm) per anchor -- lower = the input tells the probe "
          "where along the contour it is")
    print(f"{'anchor':>6s} {'kind':>11s} {'none':>6s} " + " ".join(f"{n[:9]:>9s}" for n in names))
    for gi in AI:
        m = ANC == gi
        row = f"{gi:6d} {ANCHOR_KIND[gi]:>11s} {none_al[m].mean():6.3f} "
        row += " ".join(f"{ALONG[n][m].mean():9.3f}" for n in names)
        print(row)

    print("\nWINS vs F0_base on the ALONG axis (paired over subjects, z > 3 only):")
    anywin = False
    for gi in AI:
        m = ANC == gi
        base = per_ear(ALONG["F0_base"], m)
        for nm in names:
            if nm == "F0_base":
                continue
            d = per_ear(ALONG[nm], m) - base
            z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
            if abs(z) > 3:
                anywin = True
                print(f"  anchor {gi:2d} ({ANCHOR_KIND[gi]:>11s}): {nm:14s} "
                      f"{'BETTER' if d.mean() < 0 else 'worse '} by {abs(d.mean()):.3f} mm  z={z:+.1f}")
    if not anywin:
        print("  none")

    print("\nTOTAL residual (mm), all anchors pooled:")
    print(f"  no correction {none_err.mean():.3f}")
    for nm in names:
        print(f"  {nm:14s} total {ERR[nm].mean():.3f}   along {ALONG[nm].mean():.3f}   "
              f"across {ACROSS[nm].mean():.3f}")

    # cross-fitted per-anchor selection
    chosen_err = np.zeros(len(Y)); picks = {}
    for gi in AI:
        for f in range(3):
            trm = (ANC == gi) & (fold_of != f); tem = (ANC == gi) & (fold_of == f)
            best = min(names, key=lambda n: ALONG[n][trm].mean())
            picks.setdefault(gi, []).append(best)
            chosen_err[tem] = ALONG[best][tem]
    print(f"\nper-anchor feature choice, CROSS-FITTED: along {chosen_err.mean():.3f}  "
          f"vs F0_base {ALONG['F0_base'].mean():.3f}")
    for gi in AI:
        print(f"  anchor {gi:2d} ({ANCHOR_KIND[gi]:>11s}) picks per fold: {picks[gi]}")

    np.savez_compressed(ROOT / args.out, anchor=ANC, group=GRP, fold=fold_of, y=Y, tangent=TAN,
                        **{f"err_{n}": ERR[n] for n in names},
                        **{f"along_{n}": ALONG[n] for n in names},
                        **{f"across_{n}": ACROSS[n] for n in names})
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
