"""Tangent-constrained anchor correction: sliding semilandmarks, post-hoc.

Geometric morphometrics has a mature answer to "points that are meaningful only
along a curve": SLIDING SEMILANDMARKS (Bookstein 1997; Gunz, Mitteroecker &
Bookstein 2005; Gunz & Mitteroecker 2013). Each point may move only along its
curve's tangent, to the position that best agrees with the population shape --
by minimising thin-plate-spline bending energy, or Procrustes distance, to a
reference.

That constraint is exactly our measured problem: 68% of anchor squared error is
along the contour. Constraining a shape-based correction to the tangent should
keep its along-contour benefit and remove the across-contour damage seen in
the unconstrained relational ridge (feature_probe_r2 R5(b): overall -0.030,
t = -3.69, but it made anchors 0, 22, 24, 46, 50 slightly worse).

Everything is computed from PREDICTIONS ONLY at test time (tangents from our
own predicted contour; the reference shape from TRAINING-fold ground truth),
cross-fitted over the same 3 outer folds that produced the saved predictions.
Per-anchor step size is chosen on the two training folds only.

Variants:
  V0  baseline (as shipped)
  V1  unconstrained relational ridge (R5(b) replica)
  V2  V1 projected onto the predicted tangent (along-only)
  V3  Procrustes slide: align the training mean anchor shape to our anchors,
      slide each anchor along its tangent toward it
  V4  bending-energy slide (Bookstein closed form) against the training mean
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid, kabsch
from src.foundations.splits import k_fold_subject_split
from src.interpolation.interpolation import resample_uniform

AI = list(ANCHOR_INDICES)
STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)
DATA = ROOT / "2026 Munich Tech Arena - Datas"


def contour_range(gi):
    for spec in CONTOUR_SPECS.values():
        s, e = spec["range"]
        if s <= gi < e:
            return s, e


def pred_tangents(P):
    """(n, 15, 3) unit tangents at each anchor from the PREDICTED contour."""
    T = np.zeros((len(P), len(AI), 3))
    for k, gi in enumerate(AI):
        s, e = contour_range(gi)
        lo, hi = max(gi - 1, s), min(gi + 1, e - 1)
        t = P[:, hi] - P[:, lo]
        T[:, k] = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    return T


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


def gpa_mean(shapes, iters=5):
    M = shapes[0].copy()
    for _ in range(iters):
        al = np.array([apply_rigid(s, *kabsch(s, M)) for s in shapes])
        M = al.mean(0)
    return M


def tps_bending(M):
    """3D thin-plate spline bending-energy matrix (kernel U(r) = r) for the
    k x k reference configuration M (Bookstein 1989/1997)."""
    k = len(M)
    K = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=2)
    Pm = np.hstack([np.ones((k, 1)), M])
    Lm = np.zeros((k + 4, k + 4)); Lm[:k, :k] = K; Lm[:k, k:] = Pm; Lm[k:, :k] = Pm.T
    return np.linalg.pinv(Lm)[:k, :k]


def ridge(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-8] = 1
    Xs = (X - mu) / sd; ym = Y.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(X.shape[1]), Xs.T @ (Y - ym))
    return lambda Z: ((Z - mu) / sd) @ W + ym


def main():
    d = np.load(ROOT / "results" / "twopass_surfsnap.npz", allow_pickle=True)
    pred, truth = d["pred"], d["truth"]
    subj = np.array([str(s) for s in d["subject_id"]])
    n = len(pred)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    folds = list(k_fold_subject_split(ds.subject_ids[:200], 3, 0))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(folds):
        fold_of[np.isin(subj, list(te))] = f

    PA, TA = pred[:, AI], truth[:, AI]
    TAN = pred_tangents(pred)
    proposals = {}                                    # variant -> (n,15,3) full-step target

    # V1 / V2: relational ridge, cross-fitted
    v1 = PA.copy()
    for k, gi in enumerate(AI):
        X = (PA - PA[:, k][:, None, :]).reshape(n, -1)
        for f in range(3):
            tr, te = fold_of != f, fold_of == f
            v1[te, k] = PA[te, k] + ridge(X[tr], TA[tr, k] - PA[tr, k], 100.0)(X[te])
    proposals["V1 relational ridge"] = v1
    corr = v1 - PA
    proposals["V2 ridge, tangent only"] = PA + (corr * TAN).sum(2, keepdims=True) * TAN

    # V3 / V4: slides against the TRAINING-fold mean shape
    v3 = PA.copy(); v4 = PA.copy()
    for f in range(3):
        tr, te = np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
        M = gpa_mean(TA[tr])
        Be = tps_bending(M)
        for i in te:
            Mi = apply_rigid(M, *kabsch(M, PA[i]))           # reference in this ear's pose
            v3[i] = PA[i] + ((Mi - PA[i]) * TAN[i]).sum(1, keepdims=True) * TAN[i]
            # Bookstein closed form: minimise vec(U)' (Be (x) I3) vec(U), U = P + T s
            # -> s = -(T' B T)^-1 T' B vec(P - Mi)   (energy of the deviation from Mi)
            B3 = np.kron(Be, np.eye(3))
            Tm = np.zeros((len(AI) * 3, len(AI)))
            for kk in range(len(AI)):
                Tm[kk * 3:kk * 3 + 3, kk] = TAN[i, kk]
            dev = (PA[i] - Mi).ravel()
            A = Tm.T @ B3 @ Tm + 1e-9 * np.eye(len(AI))
            s = -np.linalg.solve(A, Tm.T @ B3 @ dev)
            v4[i] = PA[i] + s[:, None] * TAN[i]
    proposals["V3 Procrustes slide"] = v3
    proposals["V4 bending-energy slide"] = v4

    base = np.linalg.norm(pred - truth, axis=2)
    print(f"V0 baseline                       overall {base.mean():.4f}   anchors {base[:, AI].mean():.4f}\n")
    print(f"{'variant':32s} {'overall':>8s} {'delta':>8s} {'t':>7s} {'anchors':>8s}   per-anchor step chosen on train folds")
    for name, full in proposals.items():
        step = full - PA
        na = pred.copy()
        chosen = np.zeros((3, len(AI)))
        for f in range(3):
            tr, te = fold_of != f, fold_of == f
            for k in range(len(AI)):
                errs = [np.linalg.norm(PA[tr, k] + w * step[tr, k] - TA[tr, k], axis=1).mean()
                        for w in STEPS]
                w = STEPS[int(np.argmin(errs))]
                chosen[f, k] = w
                na[te, AI[k]] = PA[te, k] + w * step[te, k]
        out = rebuild(pred, na)
        e = np.linalg.norm(out - truth, axis=2)
        dd = e.mean(1) - base.mean(1)
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(n))
        print(f"{name:32s} {e.mean():8.4f} {e.mean()-base.mean():+8.4f} {t:+7.2f} {e[:, AI].mean():8.4f}   "
              + " ".join(f"{gi}:{chosen[:, k].mean():.2f}" for k, gi in enumerate(AI)))
        worst = [(gi, e[:, gi].mean() - base[:, gi].mean()) for gi in AI]
        print("    per anchor: " + "  ".join(f"{gi}:{dv:+.3f}" for gi, dv in worst))


if __name__ == "__main__":
    main()
