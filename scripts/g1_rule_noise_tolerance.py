"""G1: how accurate must a contour CURVE be for the organizer's rules to beat the network?

No meshes, no training. Take the TRUE outer- and inner-helix curves, add smooth
random error of known size (the kind a curve model makes: bends, not per-point
jitter), apply the organizer's rules, and compare slider-anchor error with our
current learned anchors (T2, the relationally corrected network output).

Rules (verified on true curves earlier):
  6  = upper end of the outer helix's max-diameter chord
  22 = lowest point (z) of the outer helix
  64 = point of the inner-helix curve nearest to 6
  74 = walk 10 x the 55->64 step along the inner helix from 64
Global rules jump to a wrong chord on ~18% of true ears, so a WINDOWED version
(search only within +-8mm of arc around the network's own prediction) is also
scored -- that is how the real system would run.

Noise: per ear, per curve, a smooth 3D displacement field along arc length
(Gaussian knots every L mm, cubic interpolation), scaled to RMS sigma mm.
Calibration row: the mean distance from true landmarks to the noisy curve is
printed next to our real predicted polyline's 0.66mm on the same metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from elastic_along_curve import arc, point_at, proj_arc
from feature_probe_r3 import AI, load_sorted, rebuild

POS = {g: k for k, g in enumerate(AI)}
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]
LENGTHS = [6.0, 15.0]
WINDOW = 8.0
STEP = 0.25


def dense(poly, step=STEP):
    cum = arc(poly); s = np.arange(0.0, cum[-1] + 1e-9, step)
    return s, np.array([point_at(poly, x) for x in s])


def smooth_noise(s, sigma, L, rng):
    if sigma == 0:
        return np.zeros((len(s), 3))
    knots = np.arange(s[0] - L, s[-1] + 2 * L, L)
    field = CubicSpline(knots, rng.normal(size=(len(knots), 3)), axis=0)(s)
    return field * sigma / np.sqrt((field ** 2).sum(1).mean())


def rule6(curve, s, near=None):
    D = np.linalg.norm(curve[:, None, :] - curve[None, :, :], axis=2)
    if near is None:
        a, b = np.unravel_index(np.argmax(D), D.shape)
        return curve[a] if curve[a, 2] > curve[b, 2] else curve[b]
    s0 = proj_arc(curve, near); idx = np.flatnonzero(np.abs(s - s0) <= WINDOW)
    return curve[idx[np.argmax(D[idx].max(1))]]


def rule22(curve, s, near=None):
    if near is None:
        lower = curve[len(curve) // 3:]
        return lower[np.argmin(lower[:, 2])]
    s0 = proj_arc(curve, near); idx = np.flatnonzero(np.abs(s - s0) <= WINDOW)
    return curve[idx[np.argmin(curve[idx, 2])]]


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    n = len(pred)
    T2A = np.load(ROOT / "results" / "feature_probe_r4.npz", allow_pickle=True)["anch_T2"]
    net = {g: T2A[:, POS[g]] for g in (6, 22, 55, 64, 74)}
    err = lambda a, b: np.linalg.norm(a - b, axis=-1)
    t2full = pred.copy(); t2full[:, AI] = T2A
    t2_ear = np.linalg.norm(rebuild(pred, t2full) - truth, axis=2).mean(1)
    poly_d = np.mean([cKDTree(dense(pred[i, 0:25])[1]).query(truth[i, 0:25])[0].mean() for i in range(n)])

    print(f"{n} ears. network (T2) errors: " + "  ".join(f"{g}: {err(net[g], truth[:, g]).mean():.3f}" for g in (6, 22, 64, 74))
          + f"   | T2 overall {t2_ear.mean():.4f}")
    print(f"calibration: true landmarks -> OUR predicted outer-helix curve = {poly_d:.3f} mm\n")
    hdr = f"{'L':>4s} {'sigma':>5s} {'curve dist':>10s} {'mode':>8s} " + " ".join(f"{g:>12s}" for g in ("6", "22", "64", "74")) + "   overall vs T2 (rule used where better)"
    for tag, gs in (("true 6", (6,)), ("true 6,22,64,74", (6, 22, 64, 74))):
        full = t2full.copy(); full[:, list(gs)] = truth[:, list(gs)]
        d = np.linalg.norm(rebuild(pred, full) - truth, axis=2).mean(1) - t2_ear
        print(f"sanity: {tag:16s} overall vs T2 {d.mean():+.4f}")
    print(hdr)
    for L in LENGTHS:
        for sigma in SIGMAS:
            rng = np.random.default_rng(1000 + int(sigma * 100) + int(L))
            res = {g: np.zeros((n, 3)) for g in (6, 22, 64, 74)}
            cd = np.zeros(n)
            for i in range(n):
                so, co = dense(truth[i, 0:25]); co = co + smooth_noise(so, sigma, L, rng)
                si, ci = dense(truth[i, 55:75]); ci = ci + smooth_noise(si, sigma, L, rng)
                cd[i] = cKDTree(co).query(truth[i, 0:25])[0].mean()
                p6 = rule6(co, so, net[6][i]); p22 = rule22(co, so, net[22][i])
                s64 = proj_arc(ci, p6); s55 = proj_arc(ci, net[55][i])
                res[6][i], res[22][i] = p6, p22
                res[64][i] = point_at(ci, s64)
                res[74][i] = point_at(ci, s64 + 10 * (s64 - s55) / 9.0)
            # The rules have outlier ears (median 0.08mm but mean 0.93mm for 6 even on
            # true curves) and a wrong anchor drags ~20 in-between points with it. So a
            # rule position is used only where it agrees with the network within `gate`
            # mm, and each anchor is switched on only if that improves the OVERALL score.
            for gate in (1.0, 2.0, 3.0):
                cells, full = [], t2full.copy()
                for g in (6, 22, 64, 74):
                    agree = err(res[g], net[g]) <= gate
                    trial = full.copy(); trial[agree, g] = res[g][agree]
                    before = np.linalg.norm(rebuild(pred, full) - truth, axis=2).mean()
                    after = np.linalg.norm(rebuild(pred, trial) - truth, axis=2).mean()
                    on = after < before
                    cells.append(f"{err(trial[:, g], truth[:, g]).mean():5.2f} {100*agree.mean():3.0f}%{'+' if on else ' '}")
                    if on:
                        full = trial
                d = np.linalg.norm(rebuild(pred, full) - truth, axis=2).mean(1) - t2_ear
                print(f"{L:4.0f} {sigma:5.2f} {cd.mean():10.3f} {'gate'+str(gate):>8s} " + " ".join(f"{c:>12s}" for c in cells)
                      + f"   {d.mean():+.4f} (t {d.mean()/(d.std(ddof=1)/np.sqrt(n)+1e-12):+.1f})", flush=True)
    print("\ncells: anchor error after gating (mm), % of ears where the rule is used; '+' = switched on")


if __name__ == "__main__":
    main()
