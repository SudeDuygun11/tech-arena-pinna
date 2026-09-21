"""Are anchors at a stable FRACTION of their contour's arc length?

Budget finding: in-between points hold 82% of the error, and most of it is ALONG
the curve, inherited from anchors sliding (true path alone: 1.287 -> 1.062;
true anchors alone: -> 0.634). Anchors were always predicted from local 3D
patches or 3D coordinates. Sliding semilandmark practice (Gunz & Mitteroecker)
suggests a different descriptor: position as a fraction of the contour's arc.

1. On TRUE curves: sd of each anchor's arc fraction, in mm (sd x contour length).
   Compare with our current along-curve error for that anchor.
2. Practical test on OUR predicted curves (no truth at test time): move each
   interior anchor to (fold-wise) mean fraction of our own curve, per anchor
   only where that improves; score overall after rebuilding in-between points.
   Cross-fitted: fractions estimated on the other two folds' true curves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from elastic_along_curve import arc, proj_arc
from error_budget_paths import interp_at
from feature_probe_r3 import AI, load_sorted, rebuild
from src.foundations.contours import CONTOUR_SPECS
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split

DATA = ROOT / "2026 Munich Tech Arena - Datas"


def main():
    _, truth, subj, _, keys = load_sorted("twopass_surfsnap.npz")
    lr = np.load(ROOT / "results" / "local_refine.npz", allow_pickle=True)
    o = np.argsort(lr["keys"]); P = lr["local"][o]
    n = len(P)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    fold = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold[np.isin(subj, list(te))] = f
    err = lambda a, b: np.linalg.norm(a - b, axis=-1)
    base_ear = err(P, truth).mean(1)

    print(f"{'contour':20s} {'anchor':>6s} {'frac mean':>9s} {'sd frac':>8s} {'sd mm':>6s} {'our along err':>13s} {'our err':>8s}")
    frac_true = {}
    for name, spec in CONTOUR_SPECS.items():
        lo, hi = spec["range"]; anchors = spec["anchors"]
        L = np.array([arc(truth[i, lo:hi])[-1] for i in range(n)])
        for a in anchors[1:-1]:
            fr = np.array([arc(truth[i, lo:hi])[a - lo] for i in range(n)]) / L
            frac_true[a] = fr
            along = np.array([abs(proj_arc(truth[i, lo:hi], P[i, a]) - arc(truth[i, lo:hi])[a - lo]) for i in range(n)])
            print(f"{name:20s} {a:6d} {fr.mean():9.3f} {fr.std():8.4f} {(fr.std() * L).mean():6.2f} {along.mean():13.2f} "
                  f"{err(P[:, a], truth[:, a]).mean():8.2f}")

    print("\nmove interior anchors to the cross-fitted mean fraction of OUR curve (weight w toward it):")
    for w in (0.25, 0.5, 1.0):
        new = P.copy()
        for name, spec in CONTOUR_SPECS.items():
            lo, hi = spec["range"]
            for a in spec["anchors"][1:-1]:
                for f in range(3):
                    m = frac_true[a][fold != f].mean()
                    for i in np.flatnonzero(fold == f):
                        poly = P[i, lo:hi]; cum = arc(poly)
                        s_now = cum[a - lo]; s_prior = m * cum[-1]
                        new[i, a] = interp_at(poly, np.array([s_now + w * (s_prior - s_now)]))[0]
        R = rebuild(P, new)
        d = err(R, truth).mean(1) - base_ear
        per = {a: err(R[:, a], truth[:, a]).mean() - err(P[:, a], truth[:, a]).mean() for a in frac_true}
        print(f"  w={w:4.2f}: overall {err(R, truth).mean():.4f} ({d.mean():+.4f}, t {d.mean()/(d.std(ddof=1)/np.sqrt(n)):+.1f})  "
              + " ".join(f"{a}:{v:+.2f}" for a, v in per.items()))


if __name__ == "__main__":
    main()
