# Recipe v1 (Wing + augmentation + SWA + per-ear reference selection)

The best confirmed training recipe as of 2026-09-21. All choices were made on outer fold 0
of the 3-fold split with 3 seeds averaged (no polish stage in the screens); a full 3-fold
number is produced by the command below. Screen numbers are NOT comparable to the full
pipeline: the polish stage is worth about 0.4mm.

| step (fold 0, 3 networks averaged, no polish) | error |
|---|---|
| old baseline (SmoothL1, no augmentation, 11 typical references) | 1.712 |
| + Wing loss + rotation/scale augmentation | 1.638 |
| + SWA (`--swa-start 0.75`) | 1.630 |
| + per-ear reference selection (best 5 of 40 by surface fit) | **1.586** |

## Reproduce

```bash
pip install -r requirements.txt         # exact versions of the original run: requirements_frozen.txt
python scripts/run_full_pipeline.py --tag recipe_v1
```

Writes `results/full_run_recipe_v1/`:

| file | contents |
|---|---|
| `manifest.json` | commands, git commit, environment, seeds, fold assignment, SHA-256 of every source file, data fingerprint, timings, summary lines |
| `bundle_fold{0,1,2}.pt` | per-fold trained anchor and polish networks, template, config, exact train/test subject ids |
| `oof_predictions.npz` | out-of-fold predictions for all 400 ears |
| `bundle_final_all200.pt` | production model trained on all 200 subjects |
| `cv.log`, `final.log` | complete logs |
| `SHA256SUMS.txt` | checksums of the artifacts |

Smoke test of the runner itself (minutes): `python scripts/run_full_pipeline.py --tag smoke --smoke --out-root <dir>`.

## What is deliberately not in the recipe

Polish networks keep their legacy training (`--polish-legacy`: SmoothL1, no augmentation):
the recipe was only screened on the anchor network. Two-pass registration is off (it was a
null). Tested and rejected: curvedness/shape-index/"rich" inputs, AdamW, learned per-anchor loss
weights, per-contour networks, cosine learning rate, Gaussian curvature, reference-agreement
inputs, aligned local-frame patches, organizer construction rules. See `SOLUTION_README.md`.

## Determinism

GPU training is not bit-for-bit deterministic; seeds are fixed (`--torch-seed 0`, `--seed 0`)
and recorded. The registration cache (`REGISTRATION_CACHE_DIR`) is exact: results are identical
with or without it, it only saves time. The example cache key includes the source hashes of
registration, patch extraction and training-data code, so code changes regenerate examples.
