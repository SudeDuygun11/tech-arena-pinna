"""Where does the remaining error live -- across ears, and can cheap stacking help?

Every post-hoc correction so far lands within -0.04 to -0.05mm, and the best
(T2 ridge on all 85 predicted points, nested) reaches 1.3127mm. 1.0 needs
~0.31mm more, so the kind of method that can get there depends on the SHAPE of
the error distribution:
  * a heavy tail (a minority of badly failed ears) -> detect and repair
    failures (gross misregistration, wrong ridge, flipped fold)
  * a uniform spread -> only better base predictions for every ear will do

Also two quick stacking ideas on the saved predictions:
  A1  average the one-pass (1.3736) and two-pass (1.3653) runs
  A2  T2-style ridge correcting ALL 85 points directly (not just anchors),
      fully nested, then respaced
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from feature_probe_r2 import fit_predict
from feature_probe_r3 import AI, STEPS, load_sorted, rebuild
from src.foundations.contours import CONTOUR_SPECS
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split
from src.interpolation.interpolation import resample_uniform

DATA = ROOT / "2026 Munich Tech Arena - Datas"


def describe(name, E):
    ear = E.mean(1); s = np.sort(ear)[::-1]; n = len(s)
    top10 = s[: n // 10].sum() / s.sum(); top20 = s[: n // 5].sum() / s.sum()
    capped = np.minimum(ear, np.median(ear)).mean()
    print(f"{name:26s} mean {ear.mean():.4f}  median {np.median(ear):.4f}  p90 {np.percentile(ear, 90):.3f}  "
          f"p99 {np.percentile(ear, 99):.3f}  max {ear.max():.2f}")
    print(f"{'':26s} worst 10% of ears carry {100*top10:.1f}% of total error, worst 20% carry {100*top20:.1f}%"
          f"  | if every ear above median were AT median: {capped:.4f}")
    worst = np.argsort(ear)[::-1][: max(1, n // 10)]
    return ear, worst


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    one, _, _, _, k1 = load_sorted("tier2_trim0_iters16_kref11.npz")
    assert (keys == k1).all()
    n = len(pred)
    r4 = np.load(ROOT / "results" / "feature_probe_r4.npz", allow_pickle=True)
    assert list(r4["keys"]) == list(keys)
    t2 = pred.copy(); t2[:, AI] = r4["anch_T2"]; t2 = rebuild(pred, t2)

    E0 = np.linalg.norm(pred - truth, axis=2); E2 = np.linalg.norm(t2 - truth, axis=2)
    print("ERROR DISTRIBUTION ACROSS EARS\n")
    ear0, worst0 = describe("two-pass (1.3653)", E0)
    ear2, worst2 = describe("T2-corrected (1.3127)", E2)

    print("\n  is the worst tail concentrated on particular landmarks?  (worst 10% ears vs the rest)")
    rest = np.setdiff1d(np.arange(n), worst2)
    for name, spec in CONTOUR_SPECS.items():
        s, e = spec["range"]
        print(f"    {name:20s} worst ears {E2[worst2, s:e].mean():.3f}   others {E2[rest, s:e].mean():.3f}")
    print(f"    {'landmark 74':20s} worst ears {E2[worst2, 74].mean():.3f}   others {E2[rest, 74].mean():.3f}")
    same_subject = np.mean([np.isin(subj[i], subj[np.setdiff1d(worst2, [i])]) for i in worst2])
    print(f"  worst-10% ears whose OTHER ear is also in the worst 10%: {100*same_subject:.0f}%   "
          f"(chance ~10%)  -> subject-level failure" if same_subject > 0.2 else
          f"  worst-10% ears whose other ear is also worst-10%: {100*same_subject:.0f}% (near chance)")

    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f

    print("\nSTACKING")
    avg = 0.5 * (pred + one)
    for spec in CONTOUR_SPECS.values():
        a = spec["anchors"]
        for u, v in zip(a[:-1], a[1:]):
            for i in range(n):
                avg[i, u:v + 1] = resample_uniform(avg[i, u:v + 1])
    Ea = np.linalg.norm(avg - truth, axis=2); d = Ea.mean(1) - ear0
    print(f"  A1 average one-pass + two-pass   {Ea.mean():.4f}  vs two-pass {d.mean():+.4f} "
          f"(t {d.mean()/(d.std(ddof=1)/np.sqrt(n)):+.2f})")

    # A2: per-point ridge on all 85 points, fully nested, on top of T2's corrected anchors
    final = t2.copy()
    for f in range(3):
        tr = np.flatnonzero(fold_of != f); te = np.flatnonzero(fold_of == f)
        us = np.array(sorted(set(subj[tr]))); np.random.default_rng(f).shuffle(us)
        inner = [tr[np.isin(subj[tr], p)] for p in np.array_split(us, 2)]
        for p in range(85):
            def X(ii):
                return (t2[ii] - t2[ii, p][:, None, :]).reshape(len(ii), -1)
            Y = truth[:, p] - t2[:, p]
            oof = np.zeros((len(tr), 3)); pos = {ii: j for j, ii in enumerate(tr)}
            for j in range(2):
                a, b = inner[1 - j], inner[j]
                pb = fit_predict(X(a), Y[a], subj[a], X(b))
                for ii, v in zip(b, pb):
                    oof[pos[ii]] = v
            w = STEPS[int(np.argmin([np.linalg.norm(t2[tr, p] + s * oof - truth[tr, p], axis=1).mean() for s in STEPS]))]
            final[te, p] = t2[te, p] + w * fit_predict(X(tr), Y[tr], subj[tr], X(te))
    for spec in CONTOUR_SPECS.values():
        a = spec["anchors"]
        for u, v in zip(a[:-1], a[1:]):
            for i in range(n):
                seg = final[i, u:v + 1].copy()
                final[i, u:v + 1] = resample_uniform(seg)
                final[i, u] = seg[0]; final[i, v] = seg[-1]
    Ef = np.linalg.norm(final - truth, axis=2); d = Ef.mean(1) - ear2
    print(f"  A2 T2 + per-point ridge (all 85) {Ef.mean():.4f}  vs T2 {d.mean():+.4f} "
          f"(t {d.mean()/(d.std(ddof=1)/np.sqrt(n)):+.2f})")


if __name__ == "__main__":
    main()
