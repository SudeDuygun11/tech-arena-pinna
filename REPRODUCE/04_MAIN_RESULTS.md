# Main pipeline results

All use the REAL pipeline (trained networks). Contrast with the ceiling
measurements in `05_INTERPOLATION_TABLE.md` and `06_DIAGNOSTICS.md`.

## 1.5931mm — best validated (400 ears)

```bash
python scripts/cross_validate.py --data-dir "<parent>" \
    --n-subjects 200 --outer-folds 3 --inner-folds 4 --k-references 7 \
    --epochs 150 --with-correction --with-polish \
    --capacity large --multi-seed-ensemble --n-seeds 3
```

Reproduced independently at **1.5948mm** (with `--torch-seed 0`) — a 0.002mm
match, so this figure is stable. Per-contour, from the 1.5948 run:

| | mm |
|---|---:|
| overall | 1.5948 |
| outer_helix | 1.8278 |
| concha_outline | 1.1458 |
| inner_helix | 1.9987 |
| superior_antihelix | 1.5517 |
| **anchors only** | **1.5600** |

Split: 3 outer folds x (133 train / 67 test). Every subject tested exactly
once -> 400 ears. Runtime 3-5 h.

## 1.7834mm — the 100-subject comparator

Used as the baseline for most A/B tests, because it is half the runtime.

```bash
python scripts/cross_validate.py --data-dir "<parent>" \
    --n-subjects 100 --outer-folds 2 --inner-folds 4 --k-references 7 \
    --epochs 150 --with-correction --with-polish \
    --capacity large --multi-seed-ensemble --n-seeds 5 \
    --torch-seed 0
```

| | mm |
|---|---:|
| overall | 1.7834 |
| outer_helix | 2.1082 |
| concha_outline | 1.2809 |
| inner_helix | 2.1134 |
| superior_antihelix | 1.8192 |
| anchors only | 1.7271 |

Note this is WORSE than the 200-subject run — fewer training subjects. It is a
comparator, not a headline. Runtime ~2.5 h.

Add `--save-predictions preds.npz` to dump predictions for `error_correlation.py`.

## Historical progression (400 ears unless noted)

| stage | mm |
|---|---:|
| registration only | 3.86 |
| + anchor correction | 2.87 |
| + polish (100 subjects) | 1.98 |
| + full 200-subject scale | 1.7503 |
| + more epochs & snapshot ensembling | 1.7080 |
| + large capacity & multi-seed | **1.5931** |

## WITHDRAWN: the 1.58mm figure

Do not use it. It came from `--train-fraction 0.90`, scored on **40 ears**, not
the 400-ear CV, and changed three things at once (capacity, contour weighting,
protocol). The 90/10 split trains on more subjects and tests on 10x fewer, so
it is biased low and not comparable.

## Negative results — reproducible with these commands

| change | flag | result |
|---|---|---|
| 12mm patch radius | `--patch-radius 12.0` | **2.5606** vs 1.7834 (**+44%**) |
| geodesic patches | `--geodesic-patch` | 2.4356 fold-1 vs 1.8008 (confounded: padding 3%->17%, vertices 928->582) |
| shape prior | `--shape-prior` | −0.0072 ± 0.0164, **n.s.** (paired, 400 ears) |
| crest attraction | `--crest-attract` | +0.0005 ± 0.0057, **definitive null** (paired, 48 ears) |
