# The four things that break reproduction

Almost every failed reproduction traces to one of these, not to environment
differences.

---

## GOTCHA 1 — Diagnostic scripts use GROUND-TRUTH anchors, not predictions

`interpolation_compare.py`, `interpolation_ceiling.py`, `shape_prior_capacity.py`
and `annotation_quantization.py` feed **the true landmark positions** into the
thing being measured. No network is trained or loaded.

```python
# scripts/interpolation_compare.py
gt_anchors = {a: gt[a] for a in ANCHOR_INDICES}   # <- TRUE positions
blended = blend_correct(raw, gt_anchors, mesh.vertices)
```

**These are CEILINGS, not pipeline accuracy.** They answer "how good could this
component be if everything upstream were perfect?" If you run the real pipeline
and compare against them, nothing will match and everything will look ~0.5mm
worse.

Numbers that are ceilings: 1.3457 / 1.5405 / 1.2893 (interpolation), 0.377
(prior reconstruction), 0.534 (candidate oracle), 0.735 (prior interpolation).

Numbers that are real pipeline accuracy: **1.5931**, 1.5948, 1.7834, 2.5606.

---

## GOTCHA 2 — Different scripts score different point subsets

| script | scores |
|---|---|
| `cross_validate.py` | **all 85** points |
| `interpolation_compare.py` | **only the 70 NON-ANCHOR** points |
| `candidate_*` / `confidence_diagnostic` / `prior_strength` | **only the 15 anchors** |

Per-contour counts differ accordingly. In `interpolation_compare.py`:

| contour | total points | anchors | **scored** |
|---|---:|---:|---:|
| outer_helix | 25 | 4 | **21** |
| concha_outline | 30 | 6 | **24** |
| inner_helix | 20 | 3 | **17** |
| superior_antihelix | 10 | 2 | **8** |

Scoring all 85 in that script inflates nothing and deflates nothing predictably —
it just produces different numbers, because anchors there are exact by
construction (error 0) and would drag every contour mean down.

---

## GOTCHA 3 — `prior_snap` is a THIRD arm; raw `prior` LOSES

Three arms are reported. It is easy to compare the wrong column.

| arm | what it is | vs blend |
|---|---|---|
| `blend` | current pipeline (Stage 2) | baseline |
| `prior` | raw PCA reconstruction, no surface snap | **+0.195 ± 0.050 — significantly WORSE** |
| `prior_snap` | PCA reconstruction + the same 3mm surface snap blending uses | −0.056 ± 0.048, not significant |

The outer_helix win (1.6393 vs 2.1213) belongs to **`prior_snap`**. Raw `prior`
gets 1.9790 there. The surface snap is worth 0.25mm on its own.

---

## GOTCHA 4 — Training is nondeterministic unless `--torch-seed` is passed

`torch.manual_seed` is only called when `--torch-seed` is given. Without it,
every run initialises differently. Measured noise floor, same config run twice
at 24 subjects:

| metric | run A | run B | spread |
|---|---:|---:|---:|
| overall | 2.9970 | 3.1112 | **0.114mm** |
| superior_antihelix | 3.1124 | 3.9663 | **0.854mm** |

**Any difference under ~0.11mm at that scale is noise.** Always pass
`--torch-seed 0` for reproducibility, and vary it deliberately across repeats
when you want to measure variance.

⚠ **Do NOT pass `torch_seed` into `train_multi_seed_ensemble`.** It forwards
`**kwargs` to every member, so a fixed seed would give N identical models and
silently collapse the ensemble. `cross_validate.py` deliberately omits it there.

---

## Sanity check before debugging anything else

```bash
python scripts/cross_validate.py --data-dir "<parent>" \
    --n-subjects 6 --outer-folds 2 --inner-folds 2 --k-references 3 \
    --epochs 3 --with-correction
```

Should finish in ~1-2 minutes and report roughly 4.2-4.4mm. If that works, the
environment and data paths are fine and the problem is one of the four gotchas
above.
