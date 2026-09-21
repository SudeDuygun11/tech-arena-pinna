"""Feature research, round 3: categories that are NOT local surface descriptors.

Rounds 1-2 closed local surface features for the sliding anchors: 13 families
(normals, curvature at 3 scales, shape index, ridge direction/turning, spin and
LSP histograms, depth/fold, larger and multi-scale patches, heat and wave
kernels) -- none locates 74, 6 or 22. The only information that did was where
the OTHER landmarks are: a per-anchor relational ridge on real saved
predictions, 1.3653 -> 1.3193mm (t = -6.16). That is the bar here.

Round 3 asks which OTHER information adds to it. All tests run on the saved
out-of-fold predictions (results/twopass_surfsnap.npz), cross-fitted over the
same 3 outer subject folds that produced them, per-anchor step size chosen on
the training folds only, then re-interpolated and scored end to end.

  V1  relational ridge (the bar): other predicted anchors relative to this one
  T1  + ORGANIZER-RULE candidates evaluated on our own curves: max-diameter
        endpoints of the outer helix (6, 22), highest outer-helix point,
        projection of our 6 onto our inner helix (64), 10-step walk from our 64
        (74), inner-helix point at the height of our concha start (55), nearest
        concha point to our 75, nearest helix points to our 84
  T2  + WHOLE-SHAPE context: all 85 predicted points relative to this anchor
  T3  + MODEL DISAGREEMENT: one-pass minus two-pass prediction per anchor --
        an uncertainty signal the pipeline already produces for free
  T4  + BILATERAL: the same subject's other ear, rigidly aligned onto this ear
  T5  POSTERIOR SHAPE MODEL (Albrecht, Luthi, Gerig & Vetter, MedIA 2013):
        Gaussian conditioning of a statistical anchor-shape model on our
        predicted anchors, with the prediction-error covariance estimated on
        the training folds -- so it KNOWS our errors are mostly along the
        contour. Two noise models: per-anchor 3x3 blocks, and full 45x45.
  T6  + HEAD CONTEXT: ear position relative to the head centroid, head extent
  (vertex colour was checked first and is uninformative: 7 unique colours,
   std 0.2 -- the meshes carry no texture)

Finally, a CROSS-FITTED per-anchor choice among all variants.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from feature_probe_r2 import fit_predict                        # inner grouped CV ridge
from src.foundations.canonical import load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid, kabsch
from src.foundations.splits import k_fold_subject_split
from src.interpolation.interpolation import resample_uniform

AI = list(ANCHOR_INDICES)
KIND = {0: "junction", 6: "extremum", 22: "extremum", 24: "junction", 25: "junction",
        33: "junction", 42: "junction", 46: "junction", 50: "junction", 54: "junction",
        55: "height-ref", 64: "cross-ref", 74: "constructed", 75: "junction", 84: "junction"}
STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)
DATA = ROOT / "2026 Munich Tech Arena - Datas"


# ------------------------------------------------------------------ helpers
def load_sorted(name):
    d = np.load(ROOT / "results" / name, allow_pickle=True)
    keys = np.array([f"{s}_{x}" for s, x in zip(map(str, d["subject_id"]), map(str, d["side"]))])
    o = np.argsort(keys)
    return d["pred"][o], d["truth"][o], np.array([str(s) for s in d["subject_id"]])[o], \
        np.array([str(s) for s in d["side"]])[o], keys[o]


def densify(poly, k=30):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], len(poly) * k)
    return np.stack([np.interp(s, cum, poly[:, c]) for c in range(3)], 1), s


def rule_candidates(P):
    """Organizer-rule positions computed on OUR predicted curves (no truth)."""
    oh, _ = densify(P[0:25]); ih, s_ih = densify(P[55:75]); co, _ = densify(P[25:55])
    D = np.linalg.norm(oh[:, None, :] - oh[None, :, :], axis=2)
    a, b = np.unravel_index(np.argmax(D), D.shape)
    up, lo = (oh[a], oh[b]) if oh[a][2] > oh[b][2] else (oh[b], oh[a])
    top = oh[np.argmax(oh[:, 2])]
    c64 = ih[np.argmin(np.linalg.norm(ih - P[6], axis=1))]
    j64 = np.argmin(np.linalg.norm(ih - P[64], axis=1))
    dstep = np.linalg.norm(np.diff(P[55:65], axis=0), axis=1).mean()
    c74 = ih[np.argmin(np.abs(s_ih - min(s_ih[j64] + 10 * dstep, s_ih[-1])))]
    half = len(ih) // 2
    c55 = ih[np.argmin(np.abs(ih[:half, 2] - P[25, 2]))]
    c75 = co[np.argmin(np.linalg.norm(co - P[75], axis=1))]
    c84o = oh[np.argmin(np.linalg.norm(oh - P[84], axis=1))]
    c84i = ih[np.argmin(np.linalg.norm(ih - P[84], axis=1))]
    return np.stack([up, lo, top, c64, c74, c55, c75, c84o, c84i])       # (9, 3)


def rebuild(P, new_anchor):
    out = P.copy()
    for spec in CONTOUR_SPECS.values():
        a = spec["anchors"]
        for u, v in zip(a[:-1], a[1:]):
            su, sv = new_anchor[:, u] - P[:, u], new_anchor[:, v] - P[:, v]
            w = np.linspace(0, 1, v - u + 1)[None, :, None]
            out[:, u:v + 1] = P[:, u:v + 1] + (1 - w) * su[:, None, :] + w * sv[:, None, :]
            for i in range(len(out)):
                out[i, u:v + 1] = resample_uniform(out[i, u:v + 1])
    out[:, AI] = new_anchor[:, AI]
    return out


def head_context(subj, side, keys):
    cache = ROOT / "results" / "head_context.npz"
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        if list(c["keys"]) == list(keys):
            return c["centroid"], c["extent"]
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    cen = np.zeros((len(keys), 3)); ext = np.zeros((len(keys), 3))
    for i, (s, sd) in enumerate(zip(subj, side)):
        m = load_canonical_mesh(ds, s, sd)
        V = np.asarray(m.vertices)
        cen[i] = V.mean(0); ext[i] = np.percentile(V, 98, axis=0) - np.percentile(V, 2, axis=0)
        del m, V; gc.collect()
        if (i + 1) % 50 == 0:
            print(f"    head context {i+1}/{len(keys)}", flush=True)
    np.savez_compressed(cache, keys=keys, centroid=cen, extent=ext)
    return cen, ext


# ------------------------------------------------------------------ main
def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    one, _, _, _, keys1 = load_sorted("tier2_trim0_iters16_kref11.npz")
    assert (keys == keys1).all()
    n = len(pred)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    folds = list(k_fold_subject_split(ds.subject_ids[:200], 3, 0))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(folds):
        fold_of[np.isin(subj, list(te))] = f

    PA, TA = pred[:, AI], truth[:, AI]
    base = np.linalg.norm(pred - truth, axis=2)

    # ---------------- feature blocks (per ear, or per ear x anchor)
    print("building feature blocks...", flush=True)
    rules = np.array([rule_candidates(pred[i]) for i in range(n)])          # (n, 9, 3)
    disagree = (one[:, AI] - pred[:, AI]).reshape(n, -1)                    # (n, 45)
    other = {}
    for i in range(n):
        other.setdefault(subj[i], []).append(i)
    bil = np.zeros((n, len(AI), 3))
    for i in range(n):
        j = [k for k in other[subj[i]] if k != i]
        if j:
            bil[i] = apply_rigid(PA[j[0]], *kabsch(PA[j[0]], PA[i]))
        else:
            bil[i] = PA[i]
    print("  head context (loads each mesh once, cached)...", flush=True)
    hc, hext = head_context(subj, side, keys)

    def block(name, k):
        """Features for anchor position k (index into AI)."""
        me = PA[:, k][:, None, :]
        if name == "rel":
            return (PA - me).reshape(n, -1)
        if name == "rules":
            return (rules - me).reshape(n, -1)
        if name == "shape85":
            return (pred - me).reshape(n, -1)
        if name == "disagree":
            return disagree
        if name == "bilateral":
            return (bil - me).reshape(n, -1)
        if name == "head":
            return np.hstack([hc - PA.mean(1), hext])
        raise KeyError(name)

    RIDGE_VARIANTS = {
        "V1 relational (bar)":   ["rel"],
        "T1 + organizer rules":  ["rel", "rules"],
        "T2 + whole 85-pt shape": ["rel", "shape85"],
        "T3 + model disagreement": ["rel", "disagree"],
        "T4 + bilateral ear":    ["rel", "bilateral"],
        "T6 + head context":     ["rel", "head"],
        "T1+T3+T4 combined":     ["rel", "rules", "disagree", "bilateral"],
    }

    proposals = {}
    for name, blocks in RIDGE_VARIANTS.items():
        prop = PA.copy()
        for k in range(len(AI)):
            X = np.hstack([block(b, k) for b in blocks])
            Y = TA[:, k] - PA[:, k]
            for f in range(3):
                tr, te = fold_of != f, fold_of == f
                prop[te, k] = PA[te, k] + fit_predict(X[tr], Y[tr], subj[tr], X[te])
        proposals[name] = prop
        print(f"  fitted {name}", flush=True)

    # ---------------- T5 posterior shape model (Gaussian conditioning)
    for noise_kind in ("block", "full"):
        prop = PA.copy()
        for f in range(3):
            tr, te = np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
            # common frame: align every ear by its PREDICTED anchors (available at test)
            ref = PA[tr].mean(0)
            for _ in range(3):
                al = [kabsch(PA[i], ref) for i in tr]
                ref = np.mean([apply_rigid(PA[i], *al[t]) for t, i in enumerate(tr)], 0)
            al = {i: kabsch(PA[i], ref) for i in tr}
            Tt = np.array([apply_rigid(TA[i], *al[i]).ravel() for i in tr])
            Pt = np.array([apply_rigid(PA[i], *al[i]).ravel() for i in tr])
            mu = Tt.mean(0)
            C = np.cov(Tt.T) + 1e-3 * np.eye(45)
            Rres = np.cov((Pt - Tt).T)
            if noise_kind == "block":
                Rm = np.zeros_like(Rres)
                for k in range(len(AI)):
                    Rm[3*k:3*k+3, 3*k:3*k+3] = Rres[3*k:3*k+3, 3*k:3*k+3]
                Rres = Rm
            Rres += 1e-3 * np.eye(45)
            bias = (Pt - Tt).mean(0)
            G = C @ np.linalg.inv(C + Rres)
            for i in te:
                Rm_, tv = kabsch(PA[i], ref)
                y = apply_rigid(PA[i], Rm_, tv).ravel() - bias
                x = mu + G @ (y - mu)
                # back to this ear's pose: invert the rigid map
                xa = x.reshape(len(AI), 3)
                prop[i] = (xa - tv) @ Rm_          # inverse of apply_rigid (points @ R.T + t)
        proposals[f"T5 posterior shape ({noise_kind} noise)"] = prop
        print(f"  fitted T5 posterior shape model ({noise_kind})", flush=True)

    # ---------------- score every proposal with per-anchor step selection
    print(f"\nbaseline {base.mean():.4f}   anchors {base[:, AI].mean():.4f}\n")
    results, stepped = {}, {}
    v1_ear = None
    print(f"{'variant':34s} {'overall':>8s} {'vs base':>8s} {'t':>7s} {'vs V1':>8s} {'t':>7s}  "
          f"{'74':>6s} {'6':>6s} {'22':>6s} {'64':>6s}")
    for name, full in proposals.items():
        step = full - PA
        na = pred.copy(); per_anchor = np.zeros((n, len(AI), 3))
        for f in range(3):
            tr, te = fold_of != f, fold_of == f
            for k in range(len(AI)):
                errs = [np.linalg.norm(PA[tr, k] + w * step[tr, k] - TA[tr, k], axis=1).mean() for w in STEPS]
                w = STEPS[int(np.argmin(errs))]
                na[te, AI[k]] = PA[te, k] + w * step[te, k]
        stepped[name] = na[:, AI].copy()
        out = rebuild(pred, na)
        e = np.linalg.norm(out - truth, axis=2)
        ear = e.mean(1)
        if name.startswith("V1"):
            v1_ear = ear
        d0 = ear - base.mean(1); t0 = d0.mean() / (d0.std(ddof=1) / np.sqrt(n))
        d1 = ear - v1_ear; t1 = d1.mean() / (d1.std(ddof=1) / np.sqrt(n) + 1e-12)
        results[name] = e
        print(f"{name:34s} {e.mean():8.4f} {d0.mean():+8.4f} {t0:+7.2f} {d1.mean():+8.4f} {t1:+7.2f}  "
              f"{e[:, 74].mean():6.3f} {e[:, 6].mean():6.3f} {e[:, 22].mean():6.3f} {e[:, 64].mean():6.3f}")

    # ---------------- cross-fitted per-anchor choice among all variants
    names = list(stepped)
    choose = pred.copy(); picks = {gi: [] for gi in AI}
    for f in range(3):
        tr, te = fold_of != f, fold_of == f
        for k, gi in enumerate(AI):
            best = min(names, key=lambda nm: np.linalg.norm(stepped[nm][tr, k] - TA[tr, k], axis=1).mean())
            choose[te, gi] = stepped[best][te, k]; picks[gi].append(best.split()[0])
    out = rebuild(pred, choose)
    e = np.linalg.norm(out - truth, axis=2); ear = e.mean(1)
    d0 = ear - base.mean(1); d1 = ear - v1_ear
    print(f"\n{'PER-ANCHOR BEST (cross-fitted)':34s} {e.mean():8.4f} {d0.mean():+8.4f} "
          f"{d0.mean()/(d0.std(ddof=1)/np.sqrt(n)):+7.2f} {d1.mean():+8.4f} "
          f"{d1.mean()/(d1.std(ddof=1)/np.sqrt(n)):+7.2f}")
    for gi in AI:
        print(f"   anchor {gi:2d} ({KIND[gi]:>11s}) {base[:, gi].mean():.3f} -> {e[:, gi].mean():.3f}   picks {picks[gi]}")
    np.savez_compressed(ROOT / "results" / "feature_probe_r3.npz", keys=keys,
                        **{f"anch_{nm.split()[0]}": v for nm, v in stepped.items()},
                        best_pred=out)
    print("\nsaved -> results/feature_probe_r3.npz")


if __name__ == "__main__":
    main()
