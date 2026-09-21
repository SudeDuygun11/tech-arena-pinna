"""Converging ridge-top tracer for the outer helix and concha outline.

ridge_definition.py established the annotation tool's definition on ground
truth: outer-helix and concha contour points sit on the RIDGE TOP -- the highest
surface point across the rim along the local normal -- with 95% / 90% of true
points within 0.5mm (outer helix mean offset -0.10mm, sd 0.45). A naive one-shot
crest trace from our predicted curve landed 0.725mm from the true points (our
polyline: 0.659), because height was measured along the normal of a point that
was already off the crest, which biases the answer.

The ridge top is a FIXED POINT of "move to the highest point across the rim,
measured along the local normal". So iterate: project to the exact triangle
surface, recompute the local normal and across direction, search a shrinking
window (+-2, +-1, +-0.5mm at 0.1mm), refine the maximum sub-sample by a
parabola, repeat.

If the converged rim lands close to where the true points sit, two things follow:
  * across-contour error on these two contours can be removed by projection
  * the organizer's extremum rules (6 = max-diameter end, median 0.15mm on the
    true curve; 22 = lowest point, median 0.32mm) become better conditioned,
    and 64/74 follow from 6 by the organizer chain
Everything scored end to end against T2 (1.3127), steps chosen on inner folds.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from crest_rules import rule6, rule22
from elastic_along_curve import arc, point_at, proj_arc
from feature_probe_r3 import AI, STEPS, load_sorted, rebuild
from src.foundations.canonical import load_canonical_mesh
from src.foundations.contours import CONTOUR_SPECS
from src.foundations.dataset import Dataset
from src.foundations.geometry import nearest_surface_points
from src.foundations.splits import k_fold_subject_split

DATA = ROOT / "2026 Munich Tech Arena - Datas"
CACHE = ROOT / "results" / "crest_converged.npz"
POS = {g: k for k, g in enumerate(AI)}
TRACE = {"outer_helix": (0, 25), "concha_outline": (25, 55)}


def converge(V, F, Nv, vtree, poly, step=0.5, ext=4.0):
    cum = arc(poly); s = np.arange(-ext, cum[-1] + ext + 1e-9, step)
    pts = np.array([point_at(poly, x) for x in s])
    pts, _ = nearest_surface_points(V, F, pts)
    for half in (2.0, 1.0, 0.5):
        offs = np.arange(-half, half + 1e-9, 0.1)
        tan = np.gradient(pts, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
        _, vid = vtree.query(pts); nrm = Nv[vid]
        across = np.cross(tan, nrm); across /= np.linalg.norm(across, axis=1, keepdims=True) + 1e-12
        Q = (pts[:, None, :] + offs[None, :, None] * across[:, None, :]).reshape(-1, 3)
        S, _ = nearest_surface_points(V, F, Q)
        S = S.reshape(len(pts), len(offs), 3)
        h = ((S - pts[:, None, :]) * nrm[:, None, :]).sum(2)
        j = np.clip(np.argmax(h, 1), 1, len(offs) - 2)
        r = np.arange(len(pts))
        y0, y1, y2 = h[r, j - 1], h[r, j], h[r, j + 1]
        den = y0 - 2 * y1 + y2
        frac = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0).clip(-0.5, 0.5)
        pts = S[r, j] + frac[:, None] * (S[r, j + 1] - S[r, j - 1]) * 0.5
        pts, _ = nearest_surface_points(V, F, pts)
        sm = np.stack([np.convolve(np.pad(pts[:, c], 1, mode="edge"), np.ones(3) / 3, mode="valid") for c in range(3)], 1)
        pts = sm
    return pts


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    n = len(pred)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f

    if CACHE.exists() and list(np.load(CACHE, allow_pickle=True)["keys"]) == list(keys):
        c = np.load(CACHE, allow_pickle=True)
        tr_curves = {(i, nm): c[f"{nm}_{i}"] for i in range(n) for nm in TRACE}
    else:
        tr_curves = {}
        for i in range(n):
            m = load_canonical_mesh(ds, subj[i], side[i])
            Vall = np.asarray(m.vertices, float); Fall = np.asarray(m.faces)
            keep = np.linalg.norm(Vall - pred[i].mean(0), axis=1) <= 45.0
            sub = m.submesh([np.flatnonzero(keep[Fall].all(1))], append=True)
            del m, Vall, Fall; gc.collect()
            V = np.asarray(sub.vertices, float); F = np.asarray(sub.faces); Nv = np.asarray(sub.vertex_normals, float)
            vt = cKDTree(V)
            for nm, (a, b) in TRACE.items():
                tr_curves[(i, nm)] = converge(V, F, Nv, vt, pred[i, a:b])
            del sub, V, F, Nv, vt; gc.collect()
            if (i + 1) % 25 == 0:
                print(f"  traced {i+1}/{n}", flush=True)
        np.savez_compressed(CACHE, keys=keys, **{f"{nm}_{i}": v for (i, nm), v in tr_curves.items()})

    print("\ndistance from TRUE contour points to (mean mm):")
    for nm, (a, b) in TRACE.items():
        dp = np.mean([cKDTree(np.array([point_at(pred[i, a:b], x) for x in np.arange(0, arc(pred[i, a:b])[-1], 0.25)]))
                      .query(truth[i, a:b])[0].mean() for i in range(n)])
        dc = np.mean([cKDTree(tr_curves[(i, nm)]).query(truth[i, a:b])[0].mean() for i in range(n)])
        print(f"  {nm:16s} our polyline {dp:.3f}   converged ridge top {dc:.3f}")

    r4 = np.load(ROOT / "results" / "feature_probe_r4.npz", allow_pickle=True)
    T2A = r4["anch_T2"]
    t2full = pred.copy(); t2full[:, AI] = T2A
    t2pred = rebuild(pred, t2full)
    t2_ear = np.linalg.norm(t2pred - truth, axis=2).mean(1)
    E = lambda a, b: np.linalg.norm(a - b, axis=-1)

    c6 = np.array([rule6(tr_curves[(i, "outer_helix")]) for i in range(n)])
    c22 = np.array([rule22(tr_curves[(i, "outer_helix")]) for i in range(n)])
    print(f"\nrules on converged outer-helix ridge top:  6 {E(c6, truth[:, 6]).mean():.3f} (median {np.median(E(c6, truth[:, 6])):.3f})"
          f"   22 {E(c22, truth[:, 22]).mean():.3f} (median {np.median(E(c22, truth[:, 22])):.3f})"
          f"     T2: 6 {E(T2A[:, POS[6]], truth[:, 6]).mean():.3f}  22 {E(T2A[:, POS[22]], truth[:, 22]).mean():.3f}")

    def report(label, P):
        e = np.linalg.norm(P - truth, axis=2); d = e.mean(1) - t2_ear
        print(f"  {label:48s} {e.mean():.4f}  vs T2 {d.mean():+.4f} (t {d.mean()/(d.std(ddof=1)/np.sqrt(n)+1e-12):+.2f})"
              f"   outer {e[:, 0:25].mean():.3f}  concha {e[:, 25:55].mean():.3f}  inner {e[:, 55:75].mean():.3f}"
              f"  6 {e[:, 6].mean():.3f} 22 {e[:, 22].mean():.3f} 64 {e[:, 64].mean():.3f} 74 {e[:, 74].mean():.3f}", flush=True)

    print(f"\nend to end (T2 {t2_ear.mean():.4f}):")
    # A: project T2's outer-helix and concha points onto the converged ridge top (across correction only)
    A = t2pred.copy()
    for i in range(n):
        for nm, (a, b) in TRACE.items():
            cur = tr_curves[(i, nm)]; tree = cKDTree(cur)
            _, j = tree.query(A[i, a:b]); A[i, a:b] = cur[j]
    report("A  project onto converged ridge top", A)

    # B: A + gated, nested per-anchor rules for 6/22 + organizer chain for 64/74
    new6 = T2A[:, POS[6]].copy(); new22 = T2A[:, POS[22]].copy()
    for f in range(3):
        tr, te = fold_of != f, fold_of == f
        for g, cand, arr in ((6, c6, new6), (22, c22, new22)):
            b0 = T2A[:, POS[g]]; use = E(cand, b0) <= 3.0
            st = np.where(use[:, None], cand - b0, 0.0)
            w = STEPS[int(np.argmin([E(b0[tr] + s * st[tr], truth[tr, g]).mean() for s in STEPS]))]
            arr[te] = b0[te] + w * st[te]
            print(f"    fold {f+1} landmark {g}: step {w}", flush=True)
    anc = pred.copy(); anc[:, AI] = T2A; anc[:, 6] = new6; anc[:, 22] = new22
    for i in range(n):
        curve = pred[i, 55:75]
        s64 = proj_arc(curve, new6[i]); s55 = proj_arc(curve, T2A[i, POS[55]])
        anc[i, 64] = point_at(curve, s64); anc[i, 74] = point_at(curve, s64 + 10 * (s64 - s55) / 9.0)
    B = rebuild(pred, anc)
    for i in range(n):
        for nm, (a, b) in TRACE.items():
            cur = tr_curves[(i, nm)]; _, j = cKDTree(cur).query(B[i, a:b]); B[i, a:b] = cur[j]
    report("B  A + rules for 6/22 + chain 64/74", B)


if __name__ == "__main__":
    main()
