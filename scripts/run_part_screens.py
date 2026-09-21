"""Per-ear-part training screens, round 2b: THREE seeds per variant (resumable, sequential).

Round 2a (single seed) showed two identical baseline runs differing by 0.114mm
overall and 0.285mm on the outer helix -- as large as the effects being screened.
So every variant is now run with torch seeds 0, 1, 2 as separate runs, saved as
{name}_s{seed}.npz. The analysis averages each variant's three predictions (a
3-member ensemble of the full output, like the 1.2905 "average of runs" result)
and judges it against the 3-seed baseline with a seed-level noise estimate.
Single-seed round-2a runs are reused as the s0 (or s1/s2) members.

All runs: outer fold 0, one network per run, no polish, one-pass registration,
large capacity, 150 epochs, k_references 11. Rich/curvedness features were
dropped after round 2a (inner helix +0.20mm, 64 min per run).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "part_screens"
BASE = ["--data-dir", "2026 Munich Tech Arena - Datas", "--n-subjects", "200", "--outer-folds", "3",
        "--only-fold", "0", "--inner-folds", "4", "--k-references", "11", "--epochs", "150",
        "--with-correction", "--capacity", "large", "--multi-seed-ensemble", "--n-seeds", "1",
        "--torch-seed", "0", "--example-cache", "results/example_cache"]

WA = ["--loss", "wing", "--augment-rotate", "10", "--augment-scale", "0.1"]
VARIANTS = {
    # round 2b (done): single changes on the old baseline, then combinations
    "B0": [], "W": ["--loss", "wing"], "PC": ["--per-contour"], "G": ["--gaussian-curvature"],
    "AUG": ["--augment-rotate", "10", "--augment-scale", "0.1"],
    "WA": WA, "WAG": WA + ["--gaussian-curvature"], "WAP": WA + ["--per-contour"],
    # round 3: everything on top of the current best, Wing + augmentation (WA)
    "WA_COS": WA + ["--lr-schedule", "cosine"],
    "WA_SWA": WA + ["--swa-start", "0.75"],
    "WA_WE05": WA + ["--wing-epsilon", "0.5"],
    "WA_W2E05": WA + ["--wing-omega", "2", "--wing-epsilon", "0.5"],
    "WA_ROT20": ["--loss", "wing", "--augment-rotate", "20", "--augment-scale", "0.15"],
    "WA_DROP": WA + ["--augment-dropout", "0.3"],
    "WA_JIT": WA + ["--jitter-std", "0.75"],
    "WA_AW": WA + ["--optimizer", "adamw", "--weight-decay", "0.01"],
    "WA_UW": WA + ["--loss-weighting", "uncertainty"],
    "WA_E300": WA + ["--epochs", "300"],
    # learning curve: Wing+aug+SWA on a fraction of the training subjects, SAME number of
    # gradient steps (epochs = 150/fraction); 100% is WAS (already run, 3 seeds)
    "LC25": WA + ["--swa-start", "0.75", "--train-subject-fraction", "0.25", "--epochs", "600"],
    "LC50": WA + ["--swa-start", "0.75", "--train-subject-fraction", "0.50", "--epochs", "300"],
    "LC75": WA + ["--swa-start", "0.75", "--train-subject-fraction", "0.75", "--epochs", "200"],
    # round 4: per-ear reference selection (best 5 of 40 by surface fit) on the current best recipe
    "WAS": WA + ["--swa-start", "0.75"],
    "WAS_SEL5": WA + ["--swa-start", "0.75", "--k-references", "40", "--select-references", "5"],
    # idea 1: reference-agreement context as a side input (compare with WAS_SEL5)
    "SEL5_CTX": WA + ["--swa-start", "0.75", "--k-references", "40", "--select-references", "5",
                       "--context-features"],
    # idea 2: aligned (local-frame) anchor patches on top of selection (compare with WAS_SEL5)
    "SEL5_ALIGN": WA + ["--swa-start", "0.75", "--k-references", "40", "--select-references", "5",
                         "--aligned-patches"],
}
# cheapest and most likely first; WA_JIT regenerates examples, WA_E300 trains twice as long
ORDER = [("SEL5_ALIGN", sd) for sd in range(3)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for name, seed in ORDER:
        if only and name not in only:
            continue
        tag = f"{name}_s{seed}"
        pred_file = OUT / f"{tag}.npz"
        if pred_file.exists():
            print(f"[{tag}] already done, skipping", flush=True)
            continue
        extra = VARIANTS[name]
        args = list(BASE)
        for flag in ("--capacity", "--epochs", "--k-references"):
            if flag in extra and flag in args:
                i = args.index(flag); del args[i:i + 2]
        args[args.index("--torch-seed") + 1] = str(seed)
        cmd = [sys.executable, "-u", "scripts/cross_validate.py", *args, *extra,
               "--save-predictions", str(pred_file.relative_to(ROOT))]
        print(f"[{tag}] starting {datetime.now():%H:%M}: {' '.join(extra) or '(baseline)'}", flush=True)
        t0 = time.time()
        # exact registration cache: every run re-registers the same ears against
        # the same fold template (verified bit-identical, 3.1s -> 0.02s per ear)
        env = dict(os.environ, REGISTRATION_CACHE_DIR=str(ROOT / "results" / "registration_cache"))
        with open(OUT / f"{tag}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        print(f"[{tag}] exit {rc} after {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
