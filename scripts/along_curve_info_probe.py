"""Information test for along-curve signals, independent of sample size.

along_curve_probe.py predicted each anchor's REAL slide from along-curve windows
with ONE training example per ear per anchor (267 per fold) against ~600-2000
descriptor dims; CPR strips alone came out WORSE than baseline -- the signature
of overfitting, which cannot distinguish "the signal is not there" from "too few
examples to learn it".

This asks the information question the way rounds 1-2 did for 3D patches:
for every ear and anchor, shift the window centre along OUR predicted curve by a
known random amount delta around the TRUE anchor's projected arc position
(R shifts per ear), and ask whether each channel group recovers delta. Ten times
more examples, a known target, and the same subject-grouped folds.

Reported per anchor: mean |residual| vs the no-information baseline |delta|.
A group that brings the residual well below |delta| for the sliding anchors
(74, 6, 22, 64, 55) carries along-curve position information; one that stays at
|delta| does not, whatever the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import along_curve_probe as A
from feature_probe_r2 import fit_predict
from feature_probe_r3 import AI, KIND, load_sorted
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split

R_SHIFTS = 10
SIGMA = 2.0


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    n = len(pred)
    ds = Dataset(mesh_dir=str(A.DATA / "mesh"), landmarks_dir=str(A.DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f
    curves = {i: {ci: A.geometric_channels(pred[i], ci) for ci in range(4)} for i in range(n)}
    surf = A.surface_channels(pred, subj, side, keys, curves)          # cached
    rng = np.random.default_rng(0)
    win_u = np.arange(-A.WINDOW, A.WINDOW + 1e-9, A.DS)
    groups = {"GEO": slice(0, A.N_GEO), "SURF": slice(A.N_GEO, A.N_GEO + A.N_SURF),
              "CPR": slice(A.N_GEO + A.N_SURF, A.N_GEO + A.N_SURF + A.N_CPR)}
    combos = [("GEO",), ("SURF",), ("CPR",), ("SURF", "CPR"), ("GEO", "SURF", "CPR")]

    print(f"{'anchor':>6s} {'kind':>11s} {'|delta|':>8s} " + " ".join(f"{'+'.join(c):>13s}" for c in combos))
    pooled = {c: [] for c in combos}; pooled_none = []
    for k, gi in enumerate(AI):
        ci, s0, _ = A.contour_of(gi)
        Xw, Y, G, F = [], [], [], []
        for i in range(n):
            s, pts, cum, ch = curves[i][ci]
            ch = np.column_stack([ch, surf[(i, ci)]])
            st = s[np.argmin(np.linalg.norm(pts - truth[i, gi], axis=1))]
            for _ in range(R_SHIFTS):
                d = float(np.clip(rng.normal(0, SIGMA), -5, 5))
                Xw.append(np.stack([np.interp(st + d + win_u, s, ch[:, c_]) for c_ in range(ch.shape[1])], 1))
                Y.append(-d); G.append(subj[i]); F.append(fold_of[i])
        Xw = np.array(Xw, np.float32); Y = np.array(Y)[:, None]; G = np.array(G); F = np.array(F)
        none = np.abs(Y[:, 0]); pooled_none.append(none.mean())
        row = f"{gi:6d} {KIND[gi]:>11s} {none.mean():8.3f} "
        for c in combos:
            X = np.hstack([Xw[:, :, groups[g]].reshape(len(Xw), -1) for g in c])
            res = np.zeros(len(Y))
            for f in range(3):
                tr, te = F != f, F == f
                res[te] = np.abs(fit_predict(X[tr], Y[tr], G[tr], X[te])[:, 0] - Y[te, 0])
            pooled[c].append(res.mean())
            row += f" {res.mean():12.3f}{'*' if res.mean() < 0.7 * none.mean() else ' '}"
        print(row, flush=True)
    print(f"\n{'pooled':>18s} {np.mean(pooled_none):8.3f} " + " ".join(f"{np.mean(pooled[c]):13.3f}" for c in combos))
    print("  (* = residual below 70% of |delta|: the signal locates position along the curve)")


if __name__ == "__main__":
    main()
