# Environment

## Verified working versions

| package | version |
|---|---|
| Python | 3.13.6 |
| numpy | 2.3.2 |
| scipy | 1.16.2 |
| torch | 2.9.0+cu128 (CUDA available: True) |
| trimesh | 5.0.0 |

CUDA is used for TRAINING only. Inference runs on CPU by design — models are
moved back to CPU after training because inference is one tiny patch at a time,
where GPU kernel-launch overhead would dominate (`anchor_model.TRAIN_DEVICE`).

**A GPU is not required to reproduce any diagnostic result** — none of them
train a network. It only affects wall-clock time for `cross_validate.py`.

## Data layout

`--data-dir` must be the folder CONTAINING both of these:

```
<data-dir>/
  mesh/       P0001.ply ... P0200.ply
  landmarks/  P0001_left_ear_landmarks.csv, P0001_right_ear_landmarks.csv, ...
```

Subject order is `sorted(mesh_dir.glob("*.ply"))` — so any extra or missing
`.ply` shifts which subjects land in train vs test and changes every number.
Verify: 200 `.ply` files, 400 `.csv` files.

## Determinism

| component | deterministic? |
|---|---|
| registration (ICP + TPS) | yes — pure numpy, given the split |
| template building (GPA medoid) | yes |
| blend interpolation | yes |
| PCA shape prior | yes |
| patch point subsampling | seeded by `--seed` (numpy) |
| **network training** | **only with `--torch-seed`** |

Consequence: every diagnostic script is fully reproducible. Only
`cross_validate.py` runs that train networks need `--torch-seed`.

## Hardware notes

- Full-scale runs (200 subjects, 3 folds) take **3-5 hours**.
- Peak memory matters: `generate_examples` / `generate_polish_examples`
  explicitly `del mesh; gc.collect()` after cropping. Without that, 100+ ears
  exhausts memory and the run dies with `MemoryError` mid-`.ply` read.
- **Never run two mesh-loading jobs concurrently** — this caused three crashes
  during development.
- Disable sleep for long runs: `powercfg /change standby-timeout-ac 0`
- Watch CPU utilisation. A healthy run shows >100% (multi-core). One run was
  observed at 31% and took 11.7h of wall time for 3h39m of compute.
