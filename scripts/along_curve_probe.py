"""Along-curve information: which source tells us WHERE ALONG the contour?

WHY THIS
The error no ensemble can remove is 79% along-contour sliding (42% on the inner
helix). Every method built on our PREDICTED geometry converges to ~1.30-1.33
(relational ridge, elastic curve registration, organizer rules, multi-atlas):
they all read the same source. Yet moving each anchor to its exact position
along OUR OWN predicted curves gives 0.8795mm (t = -34) -- the remaining problem
is a one-number slide per anchor, and the missing ingredient is information
about position along the curve that the predictions do not already contain.

A clinician finds the sliding landmarks by looking ALONG the rim -- where it
stops rising, how the fold's cross-section changes, where another structure
meets it. Three sources of that, measured separately so the signal can be
attributed:

  GEO   (no mesh) curve bending, position in the ear frame, distance to each
        other predicted contour, distance to the ear centroid
  SURF  surface sampled ON the curve line: mean curvature and shape index at
        2mm, normal in the ear frame, depth from the ear plane, distance to the
        head along the inward ear normal
  CPR   curved planar reformation (Kanitsar et al., IEEE Visualization 2002 /
        2003): the surface unrolled along the curve -- at each mm of arc, the
        height of the scan surface at -8..+8mm ACROSS the curve (relative to the
        local tangent plane). The straightened cross-section strip: fold width,
        rim height, scapha depth, nearby ridges.

For each anchor: a +-WINDOW mm along-curve window centred on its predicted arc
position plus a coarse whole-contour profile. Ridge -> signed arc slide; slide
the anchor along our curve; re-interpolate; score end to end. Fully nested on
the same 3 outer folds as the saved predictions.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from feature_probe_r2 import fit_predict, principal_curvatures, shape_index
from feature_probe_r3 import AI, STEPS, load_sorted, rebuild
from src.foundations.contours import CONTOUR_SPECS
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split

DATA = ROOT / "2026 Munich Tech Arena - Datas"
WINDOW = 15.0
DS = 1.0
EXT = 12.0
N_GLOBAL = 20
CPR_OFFS = np.arange(-8.0, 8.01, 1.0)
CONTOURS = list(CONTOUR_SPECS.items())
N_GEO, N_SURF, N_CPR = 8, 7, len(CPR_OFFS)


def contour_of(gi):
    for ci, (_, spec) in enumerate(CONTOURS):
        s, e = spec["range"]
        if s <= gi < e:
            return ci, s, e


def dense_curve(poly):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.arange(-EXT, cum[-1] + EXT + 1e-9, DS)
    pts = np.stack([np.interp(np.clip(s, 0, cum[-1]), cum, poly[:, c]) for c in range(3)], 1)
    t0 = poly[1] - poly[0]; t0 /= np.linalg.norm(t0) + 1e-12
    t1 = poly[-1] - poly[-2]; t1 /= np.linalg.norm(t1) + 1e-12
    lo, hi = s < 0, s > cum[-1]
    pts[lo] = poly[0] + s[lo, None] * t0
    pts[hi] = poly[-1] + (s[hi, None] - cum[-1]) * t1
    return s, pts, cum


def ear_frame_simple(P):
    c = P.mean(0)
    return c, np.linalg.svd(P - c, full_matrices=False)[2]


def geometric_channels(P, ci):
    _, spec = CONTOURS[ci]; s0, e0 = spec["range"]
    s, pts, cum = dense_curve(P[s0:e0])
    c, B = ear_frame_simple(P)
    tang = np.gradient(pts, DS, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    turn = np.linalg.norm(np.gradient(tang, DS, axis=0), axis=1)
    others = []
    for cj, (_, sp) in enumerate(CONTOURS):
        if cj != ci:
            a, b = sp["range"]
            others.append(cKDTree(dense_curve(P[a:b])[1]).query(pts)[0])
    ch = np.column_stack([turn, (pts - c) @ B.T, np.array(others).T, np.linalg.norm(pts - c, axis=1)])
    return s, pts, cum, ch


def surface_channels(pred, subj, side, keys, curves):
    cache = ROOT / "results" / "along_curve_surface_cpr.npz"
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        if list(c["keys"]) == list(keys):
            return {(i, ci): c[f"e{i}_c{ci}"] for i in range(len(keys)) for ci in range(len(CONTOURS))}
    from src.foundations.canonical import load_canonical_mesh
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    out = {}
    print("sampling surface + CPR channels along predicted curves...", flush=True)
    for i in range(len(keys)):
        m = load_canonical_mesh(ds, subj[i], side[i])
        V = np.asarray(m.vertices, float); N = np.asarray(m.vertex_normals, float)
        del m; gc.collect()
        c, B = ear_frame_simple(pred[i]); n_out = B[2]
        if np.dot(n_out, c - V.mean(0)) < 0:
            n_out = -n_out
        tree = cKDTree(V)
        for ci in range(len(CONTOURS)):
            s, pts, cum, _ = curves[i][ci]
            _, vid = tree.query(pts)
            k1, k2, _ = principal_curvatures(V, N, tree, vid, k=48)
            ns = N[vid]
            depth = (pts - c) @ n_out
            ts = np.arange(3.0, 40.0, 0.5)
            d, j = tree.query((pts[:, None, :] - ts[None, :, None] * n_out).reshape(-1, 3))
            d = d.reshape(len(pts), -1); j = j.reshape(len(pts), -1)
            hit = (d < 0.8) & ((N[j] @ n_out) > 0.3)
            head = np.where(hit.any(1), ts[np.maximum(hit.argmax(1), 0)], 40.0)
            # curved planar reformation: surface height across the curve, per mm of arc
            tan = np.gradient(pts, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
            across = np.cross(tan, ns); across /= np.linalg.norm(across, axis=1, keepdims=True) + 1e-12
            Q = pts[:, None, :] + CPR_OFFS[None, :, None] * across[:, None, :]
            _, jq = tree.query(Q.reshape(-1, 3))
            h = ((V[jq] - Q.reshape(-1, 3)) * np.repeat(ns, len(CPR_OFFS), axis=0)).sum(1).reshape(len(pts), -1)
            out[(i, ci)] = np.column_stack([0.5 * (k1 + k2), shape_index(k1, k2), ns @ B.T, depth, head, h]).astype(np.float32)
        del V, N, tree; gc.collect()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(keys)}", flush=True)
    np.savez_compressed(cache, keys=keys, **{f"e{i}_c{ci}": v for (i, ci), v in out.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", action="store_true")
    args = ap.parse_args()

    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    n = len(pred)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f
    base = np.linalg.norm(pred - truth, axis=2)

    curves = {i: {ci: geometric_channels(pred[i], ci) for ci in range(len(CONTOURS))} for i in range(n)}
    surf = surface_channels(pred, subj, side, keys, curves) if args.mesh else None

    win_u = np.arange(-WINDOW, WINDOW + 1e-9, DS)
    W = {gi: [] for gi in AI}; Gl = {gi: [] for gi in AI}
    s_pred = np.zeros((n, len(AI))); s_true = np.zeros((n, len(AI)))
    for i in range(n):
        for k, gi in enumerate(AI):
            ci, s0, _ = contour_of(gi)
            s, pts, cum, ch = curves[i][ci]
            if surf is not None:
                ch = np.column_stack([ch, surf[(i, ci)]])
            sp = cum[gi - s0]
            s_pred[i, k] = sp
            s_true[i, k] = s[np.argmin(np.linalg.norm(pts - truth[i, gi], axis=1))]
            W[gi].append(np.stack([np.interp(sp + win_u, s, ch[:, c_]) for c_ in range(ch.shape[1])], 1))
            Gl[gi].append(np.stack([np.interp(np.linspace(s[0], s[-1], N_GLOBAL), s, ch[:, c_])
                                    for c_ in range(ch.shape[1])], 1))
    W = {gi: np.array(v) for gi, v in W.items()}; Gl = {gi: np.array(v) for gi, v in Gl.items()}
    dS = s_true - s_pred
    groups = {"GEO": slice(0, N_GEO), "SURF": slice(N_GEO, N_GEO + N_SURF),
              "CPR": slice(N_GEO + N_SURF, N_GEO + N_SURF + N_CPR)}

    def X_of(names, gi, extra_rel=False):
        parts = []
        for nm in names:
            sl = groups[nm]
            parts += [W[gi][:, :, sl].reshape(n, -1), Gl[gi][:, :, sl].reshape(n, -1)]
        parts.append(np.column_stack([s_pred[:, AI.index(gi)], [curves[i][contour_of(gi)[0]][2][-1] for i in range(n)]]))
        if extra_rel:
            parts.append((pred - pred[:, gi][:, None, :]).reshape(n, -1))
        return np.hstack(parts)

    def slide(i, k, amount):
        ci, _, _ = contour_of(AI[k]); s, pts, _, _ = curves[i][ci]
        tgt = s_pred[i, k] + amount
        return np.stack([np.interp(tgt, s, pts[:, c_]) for c_ in range(3)])

    def score(label, anchors, ref=None):
        na = pred.copy(); na[:, AI] = anchors
        e = np.linalg.norm(rebuild(pred, na) - truth, axis=2); ear = e.mean(1); d0 = ear - base.mean(1)
        msg = f"  {label:40s} {e.mean():.4f}  vs base {d0.mean():+.4f} (t {d0.mean()/(d0.std(ddof=1)/np.sqrt(n)):+.2f})"
        if ref is not None:
            d1 = ear - ref
            msg += f"  vs T2 {d1.mean():+.4f} (t {d1.mean()/(d1.std(ddof=1)/np.sqrt(n)):+.2f})"
        print(msg + f"   74 {e[:, 74].mean():.3f} 6 {e[:, 6].mean():.3f} 22 {e[:, 22].mean():.3f} "
              f"64 {e[:, 64].mean():.3f} 55 {e[:, 55].mean():.3f}", flush=True)
        return ear

    def nested_slide(names, extra_rel=False):
        out = pred[:, AI].copy()
        Xs = {gi: X_of(names, gi, extra_rel) for gi in AI}
        for f in range(3):
            tr = np.flatnonzero(fold_of != f); te = np.flatnonzero(fold_of == f)
            us = np.array(sorted(set(subj[tr]))); np.random.default_rng(f).shuffle(us)
            inner = [tr[np.isin(subj[tr], p)] for p in np.array_split(us, 2)]
            for k, gi in enumerate(AI):
                Xk = Xs[gi]; y = dS[:, k:k + 1]
                oof = np.zeros(len(tr)); pos = {ii: j for j, ii in enumerate(tr)}
                for j in range(2):
                    a, b = inner[1 - j], inner[j]
                    for ii, v in zip(b, fit_predict(Xk[a], y[a], subj[a], Xk[b])[:, 0]):
                        oof[pos[ii]] = v
                w = STEPS[int(np.argmin([np.mean([np.linalg.norm(slide(ii, k, s_ * oof[pos[ii]]) - truth[ii, gi])
                                                  for ii in tr]) for s_ in STEPS]))]
                for ii, v in zip(te, fit_predict(Xk[tr], y[tr], subj[tr], Xk[te])[:, 0]):
                    out[ii, k] = slide(ii, k, w * v)
        return out

    print(f"\nbaseline {base.mean():.4f}")
    oracle = np.array([[slide(i, k, dS[i, k]) for k in range(len(AI))] for i in range(n)])
    score("CEILING exact slides on our curves", oracle)
    r4 = np.load(ROOT / "results" / "feature_probe_r4.npz", allow_pickle=True)
    t2_ear = score("T2 relational (reference)", r4["anch_T2"])
    runs = [("GEO",), ] if surf is None else [("GEO",), ("SURF",), ("CPR",), ("SURF", "CPR"), ("GEO", "SURF", "CPR")]
    saved = {}
    for names in runs:
        A = nested_slide(list(names)); saved["+".join(names)] = A
        score(f"slide from {'+'.join(names)}", A, t2_ear)
    if surf is not None:
        A = nested_slide(["GEO", "SURF", "CPR"], extra_rel=True); saved["ALL+REL"] = A
        score("slide from GEO+SURF+CPR + all-85 relational", A, t2_ear)
    np.savez_compressed(ROOT / "results" / "along_curve_probe.npz", keys=keys, **saved)


if __name__ == "__main__":
    main()
