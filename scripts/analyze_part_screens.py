"""Score 3-seed training screens per ear part and per anchor.

Files are results/part_screens/{variant}_s{seed}.npz. For each variant:
  * ens  = error of the seed-AVERAGED prediction (a 3-member ensemble)
  * seed-level test: mean of per-seed errors vs baseline per-seed errors, with a
    Welch t over seeds -- this is the honest test, because it includes training
    randomness. The per-ear paired t is also shown but it IGNORES seed noise
    (two identical baseline seeds differ at per-ear t = 3.8).
A variant/part is marked * only when the seed-level |t| > 2.5 with >= 2 seeds
on both sides. Winners chosen on fold 0 must be confirmed on folds 1-2.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS

OUT = ROOT / "results" / "part_screens"
AI = list(ANCHOR_INDICES)
PARTS = {"overall": np.arange(85), **{c: np.arange(*s["range"]) for c, s in CONTOUR_SPECS.items()}}


def load(path):
    d = np.load(path, allow_pickle=True)
    k = np.array([f"{s}_{x}" for s, x in zip(map(str, d["subject_id"]), map(str, d["side"]))]); o = np.argsort(k)
    return k[o], d["pred"][o], d["truth"][o]


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b) + 1e-12)


def main():
    runs = defaultdict(dict)
    for p in sorted(OUT.glob("*_s[0-9].npz")):
        m = re.match(r"(.+)_s(\d)$", p.stem)
        runs[m.group(1)][int(m.group(2))] = p
    if "B0" not in runs:
        print("no baseline yet"); return
    k0 = truth = None
    P = defaultdict(dict)
    for v, seeds in runs.items():
        for s, path in seeds.items():
            k, p, t = load(path)
            if k0 is None:
                k0, truth = k, t
            assert (k == k0).all() and np.abs(t - truth).max() == 0, path
            P[v][s] = p
    err = lambda pred: np.linalg.norm(pred - truth, axis=2)  # (ears, 85)
    ens = {v: err(np.mean(list(P[v].values()), axis=0)) for v in P}
    # Ensembling alone is a big effect (2 baseline seeds averaged: -0.11mm), so a
    # variant is compared with the baseline ensembled over the SAME number of seeds,
    # averaged over every baseline seed subset of that size.
    from itertools import combinations

    def base_at(n):
        subsets = list(combinations(sorted(P["B0"]), min(n, len(P["B0"]))))
        return np.mean([err(np.mean([P["B0"][s] for s in c], axis=0)) for c in subsets], axis=0)
    per_seed = {v: {s: err(p) for s, p in P[v].items()} for v in P}
    B = "B0"
    tt = lambda d: d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)

    print(f"{len(k0)} held-out ears (fold 0). seeds per variant: "
          + "  ".join(f"{v}:{len(P[v])}" for v in P))
    print("\nENSEMBLE of available seeds (mm), change vs baseline ensemble, seed-level t:")
    print(f"{'variant':>8s} " + "".join(f"{pn[:18]:>28s}" for pn in PARTS))
    for v in [B] + [x for x in P if x != B]:
        line = f"{v:>8s} "
        for pn, idx in PARTS.items():
            e = ens[v][:, idx].mean()
            if v == B:
                sd = np.std([per_seed[B][s][:, idx].mean() for s in per_seed[B]], ddof=1) if len(P[B]) > 1 else np.nan
                line += f"{e:12.4f} seed-sd {sd:.3f}   "
                continue
            d = e - base_at(len(P[v]))[:, idx].mean()
            ts = welch([per_seed[v][s][:, idx].mean() for s in per_seed[v]],
                       [per_seed[B][s][:, idx].mean() for s in per_seed[B]])
            star = "*" if np.isfinite(ts) and abs(ts) > 2.5 else " "
            line += f"{e:9.4f} {d:+.3f}{star}(ts{ts:+5.1f})   "
        print(line)
    print("  * = seed-level |t| > 2.5 (needs >= 2 seeds both sides)")

    print("\nbaseline vs number of seeds averaged: "
          + "  ".join(f"{n} seed(s) {base_at(n).mean():.4f}" for n in range(1, len(P[B]) + 1)))
    print("\nper anchor, ensemble error (mm):")
    names = [x for x in P if x != B]
    print(f"{'anchor':>6s} {'B0':>6s} " + " ".join(f"{n:>6s}" for n in names))
    for g in AI:
        print(f"{g:6d} {ens[B][:, g].mean():6.3f} " + " ".join(f"{ens[n][:, g].mean():6.3f}" for n in names))

    print("\nper-part best ensemble (confirm on folds 1-2):")
    for pn, idx in list(PARTS.items())[1:]:
        if not names:
            break
        b = min(names, key=lambda n: ens[n][:, idx].mean() - base_at(len(P[n]))[:, idx].mean())
        bb = base_at(len(P[b]))
        d = ens[b][:, idx].mean(1) - bb[:, idx].mean(1)
        print(f"  {pn:20s} {b:>5s} {ens[b][:, idx].mean():.4f} vs B0@{len(P[b])} seeds {bb[:, idx].mean():.4f} "
              f"({d.mean():+.3f}; per-ear t {tt(d):+.2f}; seeds {len(P[b])} vs {len(P[B])})")


if __name__ == "__main__":
    main()
