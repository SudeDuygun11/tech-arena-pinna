"""Organizer extremum rules on the RIM CREST traced from the scan surface.

What the evidence reduced the problem to
  * the inner-helix error is INHERITED: with the organizer's rules applied on
    our own predicted inner-helix curve (64 = projection of 6, 74 = ten steps at
    the 55->64 spacing), a correct landmark 6 alone takes the overall from
    1.3127 to 1.2378 (true 6 + true 55: 1.1930)
  * landmark 6 = upper end of the outer helix's max-diameter chord (median
    0.047mm on true curves); landmark 22 = lowest point of the outer helix in
    the head vertical axis (median 0.110mm on true curves)
  * both are EXTREMA of a flat rim, so their position along it is set by the
    rim's exact local direction -- which our 25-point predicted polyline gets
    slightly wrong (rule on our polyline: 6 at 1.754mm, worse than learned)
  * the organizer's definitions say both are "annotated on the top of the ridge"

So: trace the actual rim CREST on the scan near our predicted outer helix (our
curve is within ~0.65mm of the true path, so the search window is safe), apply
the extremum rules to that crest, then chain 64 and 74 by the organizer rules.

Crest tracing, per 0.5mm of arc along our predicted outer helix: sample the
surface across the curve at -4..+4mm (0.25mm steps); height of the surface
along the local normal; crest = the across-offset of maximum height; light
smoothing along the arc.

Scored: rule candidates vs truth; then end to end with a per-anchor blend
between T2 and the rule candidate (step chosen on inner folds) and a gate that
ignores candidates far from T2 (the rules have outlier ears).
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from elastic_along_curve import arc, point_at, proj_arc
from feature_probe_r3 import AI, STEPS, load_sorted, rebuild
from src.foundations.canonical import load_canonical_mesh
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split

DATA = ROOT / "2026 Munich Tech Arena - Datas"
CACHE = ROOT / "results" / "crest_outer_helix.npz"
POS = {g: k for k, g in enumerate(AI)}


def dense(poly, step=0.5, ext=8.0):
    cum = arc(poly); s = np.arange(-ext, cum[-1] + ext + 1e-9, step)
    return s, np.array([point_at(poly, x) for x in s])


def trace_crest(V, N, tree, poly):
    s, pts = dense(poly)
    _, vid = tree.query(pts); nrm = N[vid]
    tan = np.gradient(pts, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    across = np.cross(tan, nrm); across /= np.linalg.norm(across, axis=1, keepdims=True) + 1e-12
    offs = np.arange(-4.0, 4.01, 0.25)
    Q = pts[:, None, :] + offs[None, :, None] * across[:, None, :]
    _, jq = tree.query(Q.reshape(-1, 3))
    Vq = V[jq].reshape(len(pts), len(offs), 3)
    h = ((Vq - pts[:, None, :]) * nrm[:, None, :]).sum(2)
    best = np.argmax(h, axis=1)
    crest = Vq[np.arange(len(pts)), best]
    ker = np.ones(5) / 5
    sm = np.stack([np.convolve(np.pad(crest[:, c], 2, mode="edge"), ker, mode="valid") for c in range(3)], 1)
    return sm


def rule6(curve):
    D = np.linalg.norm(curve[:, None, :] - curve[None, :, :], axis=2)
    a, b = np.unravel_index(np.argmax(D), D.shape)
    return curve[a] if curve[a, 2] > curve[b, 2] else curve[b]


def rule22(curve):
    lower = curve[len(curve) // 3:]
    return lower[np.argmin(lower[:, 2])]


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    n = len(pred)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f

    if CACHE.exists() and list(np.load(CACHE, allow_pickle=True)["keys"]) == list(keys):
        c = np.load(CACHE, allow_pickle=True); crests = [c[f"c{i}"] for i in range(n)]
    else:
        crests = []
        for i in range(n):
            m = load_canonical_mesh(ds, subj[i], side[i])
            V = np.asarray(m.vertices, float); N = np.asarray(m.vertex_normals, float); del m; gc.collect()
            crests.append(trace_crest(V, N, cKDTree(V), pred[i, 0:25]))
            del V, N; gc.collect()
            if (i + 1) % 50 == 0:
                print(f"  crest traced {i+1}/{n}", flush=True)
        np.savez_compressed(CACHE, keys=keys, **{f"c{i}": c for i, c in enumerate(crests)})

    r4 = np.load(ROOT / "results" / "feature_probe_r4.npz", allow_pickle=True)
    T2A = r4["anch_T2"]
    err = lambda a, b: np.linalg.norm(a - b, axis=-1)
    cand6_crest = np.array([rule6(c) for c in crests]); cand22_crest = np.array([rule22(c) for c in crests])
    poly_dense = [dense(pred[i, 0:25])[1] for i in range(n)]
    cand6_poly = np.array([rule6(c) for c in poly_dense]); cand22_poly = np.array([rule22(c) for c in poly_dense])
    true_dense = [dense(truth[i, 0:25], ext=0.0)[1] for i in range(n)]
    cand6_true = np.array([rule6(c) for c in true_dense]); cand22_true = np.array([rule22(c) for c in true_dense])

    d_crest = np.array([cKDTree(crests[i]).query(truth[i, 0:25])[0].mean() for i in range(n)])
    d_poly = np.array([cKDTree(poly_dense[i]).query(truth[i, 0:25])[0].mean() for i in range(n)])
    print(f"distance of true outer-helix landmarks to: our polyline {d_poly.mean():.3f} mm   traced crest {d_crest.mean():.3f} mm\n")
    print(f"{'landmark':>9s} {'ours':>7s} {'T2':>7s} {'rule/poly':>10s} {'rule/crest':>11s} {'rule/TRUE':>10s}   (mean | median)")
    for g, cp, cc, ct in [(6, cand6_poly, cand6_crest, cand6_true), (22, cand22_poly, cand22_crest, cand22_true)]:
        row = f"{g:9d} {err(pred[:, g], truth[:, g]).mean():7.3f} {err(T2A[:, POS[g]], truth[:, g]).mean():7.3f}"
        for cand in (cp, cc, ct):
            e = err(cand, truth[:, g]); row += f"  {e.mean():5.2f}|{np.median(e):4.2f}"
        print(row)

    base = np.linalg.norm(pred - truth, axis=2)
    t2full = pred.copy(); t2full[:, AI] = T2A
    t2_ear = np.linalg.norm(rebuild(pred, t2full) - truth, axis=2).mean(1)

    def end_to_end(label, c6, c22, gate):
        """Nested per-anchor blend T2 -> candidate (gated), then organizer chain for 64/74."""
        new6 = T2A[:, POS[6]].copy(); new22 = T2A[:, POS[22]].copy()
        for f in range(3):
            tr = fold_of != f; te = fold_of == f
            for g, cand, arr in ((6, c6, new6), (22, c22, new22)):
                base_a = T2A[:, POS[g]]
                use = err(cand, base_a) <= gate
                step = np.where(use[:, None], cand - base_a, 0.0)
                w = STEPS[int(np.argmin([err(base_a[tr] + s * step[tr], truth[tr, g]).mean() for s in STEPS]))]
                arr[te] = base_a[te] + w * step[te]
        new = pred.copy(); new[:, AI] = T2A; new[:, 6] = new6; new[:, 22] = new22
        for i in range(n):
            curve = pred[i, 55:75]
            s64 = proj_arc(curve, new6[i]); s55 = proj_arc(curve, T2A[i, POS[55]])
            new[i, 64] = point_at(curve, s64); new[i, 74] = point_at(curve, s64 + 10 * (s64 - s55) / 9.0)
        chained = rebuild(pred, new)
        # also the version WITHOUT chaining 64/74, to separate the two effects
        nochain = pred.copy(); nochain[:, AI] = T2A; nochain[:, 6] = new6; nochain[:, 22] = new22
        for tag, P in (("6/22 only", rebuild(pred, nochain)), ("6/22 + chain 64,74", chained)):
            e = np.linalg.norm(P - truth, axis=2); d = e.mean(1) - t2_ear
            print(f"  {label:26s} {tag:20s} {e.mean():.4f}  vs T2 {d.mean():+.4f} "
                  f"(t {d.mean()/(d.std(ddof=1)/np.sqrt(n)):+.2f})   6 {e[:, 6].mean():.3f} 22 {e[:, 22].mean():.3f} "
                  f"64 {e[:, 64].mean():.3f} 74 {e[:, 74].mean():.3f}", flush=True)

    print(f"\nend to end (baseline {base.mean():.4f}, T2 {t2_ear.mean():.4f}):")
    for gate in (1.5, 3.0):
        end_to_end(f"crest rules, gate {gate}mm", cand6_crest, cand22_crest, gate)
    end_to_end("polyline rules, gate 3mm", cand6_poly, cand22_poly, 3.0)


if __name__ == "__main__":
    main()
