"""Per-part specialisation, tested post-hoc before any training is spent.

User's principle: every part of the ear has different geometry, depth and
curvature, so (1) each landmark may need different input features -- a change
that helps one can hurt another -- and (2) once the anchors are good, each
anchor-to-anchor segment may deserve its own interpolation rule, rather than
one global rule for all 14 segments.

Four checks, all on saved predictions, all leakage-free (anything LEARNED is
fitted on two outer folds and applied to the third, using the same subject
split as cross_validate.py: k_fold_subject_split(subject_ids[:200], 3, seed=0)).

  E1  Is the organizer's "uniform spacing" rule actually uniform in EVERY
      segment of the ground truth? Per-segment gap CV.
  E2  Learned per-segment spacing: replace uniform arc-length placement with
      each segment's mean ground-truth arc FRACTIONS (fitted on train folds).
      Applied to (a) our predictions and (b) oracle anchors -- because the
      value of any interpolation rule depends on anchor quality.
  E3  With perfect anchors, is the remaining interpolation error SPACING
      (along the contour) or PATH SHAPE (across it)? Decides what kind of
      per-segment rule is worth building.
  E4  Per-landmark model selection, the core of the feature-study design:
      two existing runs (one-pass 1.3736, two-pass 1.3653) as stand-ins for
      two feature sets. Choose the better source per landmark on two folds,
      apply to the third. Compared with choosing on the test fold itself, to
      show how much selection bias a naive per-landmark study would report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split
from src.interpolation.interpolation import resample_uniform

DATA = ROOT / "2026 Munich Tech Arena - Datas"
AI = list(ANCHOR_INDICES)


def load(name):
    d = np.load(ROOT / "results" / name, allow_pickle=True)
    keys = [f"{s}_{side}" for s, side in zip(map(str, d["subject_id"]), map(str, d["side"]))]
    order = np.argsort(keys)
    return (d["pred"][order], d["truth"][order],
            np.array([str(s) for s in d["subject_id"]])[order], np.array(keys)[order])


def segments():
    for name, spec in CONTOUR_SPECS.items():
        a = spec["anchors"]
        for u, v in zip(a[:-1], a[1:]):
            if v - u >= 2:
                yield name, u, v


def arc_fractions(poly):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum / max(cum[-1], 1e-12), cum


def resample_at_fractions(poly, fr):
    _, cum = arc_fractions(poly)
    want = fr * cum[-1]
    return np.stack([np.interp(want, cum, poly[:, k]) for k in range(3)], axis=1)


def oracle_anchor_version(P, T):
    out = P.copy()
    for spec in CONTOUR_SPECS.values():
        a = spec["anchors"]
        for u, v in zip(a[:-1], a[1:]):
            su, sv = T[:, u] - P[:, u], T[:, v] - P[:, v]
            w = np.linspace(0, 1, v - u + 1)[None, :, None]
            out[:, u:v+1] = P[:, u:v+1] + (1-w)*su[:, None, :] + w*sv[:, None, :]
            for i in range(len(P)):
                out[i, u:v+1] = resample_uniform(out[i, u:v+1])
    out[:, AI] = T[:, AI]
    return out


def main():
    A, T, subj, keys = load("tier2_trim0_iters16_kref11.npz")   # one-pass
    B, TB, _, keysB = load("twopass_surfsnap.npz")               # two-pass
    assert (keys == keysB).all() and np.abs(T - TB).max() == 0
    n = len(A)
    err = lambda P, idx=slice(None): float(np.linalg.norm(P - T, axis=2)[:, idx].mean())

    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    folds = list(k_fold_subject_split(ds.subject_ids[:200], 3, 0))
    fold_of = np.zeros(n, int)
    for f, (_, test_ids) in enumerate(folds):
        fold_of[np.isin(subj, list(test_ids))] = f
    print(f"{n} ears; fold sizes {np.bincount(fold_of)}\n")

    # ---------------------------------------------------------------- E1
    print("E1  ground-truth gap CV WITHIN each anchor segment (0% = perfectly uniform)")
    for name, u, v in segments():
        g = np.linalg.norm(np.diff(T[:, u:v+1], axis=1), axis=2)
        cv = (g.std(1) / g.mean(1)).mean()
        print(f"    {name:19s} {u:2d}->{v:2d} ({v-u:2d} gaps)  CV {100*cv:5.2f}%")

    # ---------------------------------------------------------------- E2
    print("\nE2  learned per-segment spacing (fit on 2 folds, apply to the 3rd)")
    O = oracle_anchor_version(B, T)
    for label, P in (("our predictions (two-pass)", B), ("ORACLE anchors", O)):
        new = P.copy()
        for name, u, v in segments():
            for f in range(3):
                tr, te = fold_of != f, np.flatnonzero(fold_of == f)
                F = np.mean([arc_fractions(T[i, u:v+1])[0] for i in np.flatnonzero(tr)], axis=0)
                for i in te:
                    new[i, u:v+1] = resample_at_fractions(P[i, u:v+1], F)
        print(f"    {label:28s} uniform {err(P):.4f} -> learned {err(new):.4f} "
              f"({err(new)-err(P):+.4f})")

    # biggest systematic deviations from uniform, in mm
    print("    largest mean deviation of GT spacing from uniform, per segment:")
    rows = []
    for name, u, v in segments():
        F = np.mean([arc_fractions(T[i, u:v+1])[0] for i in range(n)], axis=0)
        L = np.mean([arc_fractions(T[i, u:v+1])[1][-1] for i in range(n)])
        rows.append((np.abs(F - np.linspace(0, 1, v-u+1)).max() * L, name, u, v))
    for mm, name, u, v in sorted(rows, reverse=True)[:5]:
        print(f"      {name:19s} {u:2d}->{v:2d}  {mm:.3f} mm")

    # ---------------------------------------------------------------- E3
    print("\nE3  with ORACLE anchors: remaining interior error, along vs across")
    interior = np.ones(85, bool); interior[AI] = False
    for name, spec in CONTOUR_SPECS.items():
        s, e = spec["range"]
        al, ac = [], []
        for gi in range(s, e):
            if not interior[gi]:
                continue
            lo, hi = max(gi-1, s), min(gi+1, e-1)
            t = T[:, hi] - T[:, lo]; t /= np.linalg.norm(t, axis=1, keepdims=True)
            d = O[:, gi] - T[:, gi]
            a = (d * t).sum(1)
            al.append(a**2); ac.append((d**2).sum(1) - a**2)
        al, ac = np.mean(al), np.mean(ac)
        print(f"    {name:19s} {100*al/(al+ac):5.1f}% along (spacing)   "
              f"{100*ac/(al+ac):5.1f}% across (path shape)")

    # ---------------------------------------------------------------- E4
    print("\nE4  per-landmark selection between two runs (stand-in for two feature sets)")
    eA = np.linalg.norm(A - T, axis=2); eB = np.linalg.norm(B - T, axis=2)
    cross = np.zeros_like(eA); naive = np.minimum(eA.mean(0), eB.mean(0)).mean()
    wins = 0
    for f in range(3):
        tr, te = fold_of != f, fold_of == f
        pickB = eB[tr].mean(0) < eA[tr].mean(0)
        cross[te] = np.where(pickB, eB[te], eA[te])
        wins += pickB.sum()
    print(f"    one-pass everywhere            {eA.mean():.4f}")
    print(f"    two-pass everywhere            {eB.mean():.4f}")
    print(f"    per-landmark, CROSS-FITTED     {cross.mean():.4f}   <- honest")
    print(f"    per-landmark, chosen ON TEST   {naive:.4f}   <- selection-biased")
    print(f"    two-pass chosen for {wins/3:.1f} of 85 landmarks on average")

    # which landmarks disagree strongly (a real conflict, not noise)?
    se = np.sqrt(((eA - eB)).var(0, ddof=1) / n)
    z = (eA - eB).mean(0) / se
    print("    landmarks where the runs genuinely conflict (|z| > 3):")
    for k in np.argsort(-np.abs(z))[:8]:
        if abs(z[k]) > 3:
            better = "two-pass" if z[k] > 0 else "one-pass"
            print(f"      lm {k:2d}  {better} better by {abs((eA-eB)[:, k].mean()):.3f} mm  (z={z[k]:+.1f})")


if __name__ == "__main__":
    main()
