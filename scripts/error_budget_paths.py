"""Where does the remaining error of the best pipeline (1.2822) live? No meshes, no training.

Every recent effort attacked the 15 anchors. This splits the error into
  anchors vs the 70 in-between points, and for in-between points:
    ACROSS  = distance to the true curve (wrong path)
    ALONG   = the rest (right path, wrong position along it)
and runs three oracles on the saved predictions:
  A  true anchors, our path                 -> what anchors alone can still buy
  B  our anchors slid onto the true curve,
     in-between points spaced along TRUE path -> what a perfect path can buy
  D  true anchors + true path + our uniform
     spacing rule                           -> error of the SPACING RULE itself
If D is far from 0 on some contour, the organizer's redistribution is not plain
uniform arc length there -- a free, deterministic fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from elastic_along_curve import arc, point_at, proj_arc
from feature_probe_r3 import AI, load_sorted, rebuild
from src.foundations.contours import CONTOUR_SPECS
from src.interpolation.interpolation import resample_uniform


def interp_at(poly, ss):
    """Vectorised point_at for many arc positions (clamped to the curve)."""
    cum = arc(poly); ss = np.clip(ss, 0, cum[-1])
    return np.stack([np.interp(ss, cum, poly[:, c]) for c in range(3)], 1)


def dense(poly, step=0.05):
    cum = arc(poly); s = np.arange(0.0, cum[-1] + 1e-9, step)
    return s, interp_at(poly, s)


def main():
    _, truth, _, _, keys = load_sorted("twopass_surfsnap.npz")
    lr = np.load(ROOT / "results" / "local_refine.npz", allow_pickle=True)
    o = np.argsort(lr["keys"]); assert list(lr["keys"][o]) == list(keys)
    P = lr["local"][o]
    n = len(P)
    E = np.linalg.norm(P - truth, axis=2)
    is_anchor = np.zeros(85, bool); is_anchor[AI] = True
    print(f"best pipeline: {E.mean():.4f} mm over {n} ears")
    print(f"  anchors (15): mean {E[:, is_anchor].mean():.3f}  -> share of total error {E[:, is_anchor].sum() / E.sum():.0%}")
    print(f"  in-between (70): mean {E[:, ~is_anchor].mean():.3f}  -> share {E[:, ~is_anchor].sum() / E.sum():.0%}\n")

    A = rebuild(P, np.where(is_anchor[None, :, None], truth, P))
    B = P.copy(); D = truth.copy()
    across = np.zeros((n, 85))
    for spec in CONTOUR_SPECS.values():
        lo, hi = spec["range"]; anchors = spec["anchors"]
        for i in range(n):
            tpoly = truth[i, lo:hi]
            s_true, dcurve = dense(tpoly)
            tree = cKDTree(dcurve)
            across[i, lo:hi] = tree.query(P[i, lo:hi])[0]
            # B: our anchors' positions ALONG the true curve, path from truth
            s_anchor = {a: proj_arc(tpoly, P[i, a]) for a in anchors}
            for u, v in zip(anchors[:-1], anchors[1:]):
                ss = np.linspace(s_anchor[u], s_anchor[v], v - u + 1)
                B[i, u:v + 1] = interp_at(tpoly, ss)
            B[i, anchors] = P[i, anchors]
            # D: true anchors + true path (the true polyline) + uniform spacing
            for u, v in zip(anchors[:-1], anchors[1:]):
                D[i, u:v + 1] = resample_uniform(truth[i, u:v + 1])
    EA, EB, ED = (np.linalg.norm(X - truth, axis=2) for X in (A, B, D))
    along = np.sqrt(np.clip(E ** 2 - across ** 2, 0, None))

    print(f"{'contour':20s} {'now':>6s} {'across':>7s} {'along':>6s}  {'A true anchors':>15s} {'B true path':>12s} {'D spacing rule':>15s}")
    for name, spec in CONTOUR_SPECS.items():
        lo, hi = spec["range"]; mid = [k for k in range(lo, hi) if not is_anchor[k]]
        print(f"{name:20s} {E[:, lo:hi].mean():6.3f} {across[:, mid].mean():7.3f} {along[:, mid].mean():6.3f}  "
              f"{EA[:, lo:hi].mean():15.3f} {EB[:, lo:hi].mean():12.3f} {ED[:, lo:hi].mean():15.3f}")
    print(f"{'OVERALL':20s} {E.mean():6.3f} {across[:, ~is_anchor].mean():7.3f} {along[:, ~is_anchor].mean():6.3f}  "
          f"{EA.mean():15.3f} {EB.mean():12.3f} {ED.mean():15.3f}")
    print("\n(across/along: in-between points only; A/B/D: all points of the contour)")


if __name__ == "__main__":
    main()
