# Reproducing the interpolation table (esp. the BLEND column)

The numbers in question:

| contour | **blend** | prior_snap |
|---|---:|---:|
| outer_helix (21 pts) | **2.1213** | 1.6393 |
| concha_outline (24) | **1.0455** | 1.1292 |
| inner_helix (17) | **1.1073** | 1.2230 |
| superior_antihelix (8) | **0.7169** | 0.9917 |
| **overall (70 pts)** | **1.3457** | 1.2893 |

---

## The command

```bash
cd Example_notebook_data
python scripts/interpolation_compare.py \
    --data-dir "2026 Munich Tech Arena - Datas" \
    --n-subjects 60
```

Every other parameter is a default and must NOT be changed:
`--k-references 7`, `--seed 0`, `N_MODES = 45`, `RIDGE = 0.05`.

Runtime ~20 min. No GPU needed — **no network is trained or loaded.**

---

## What "blend" is, exactly

The blend column is `src/interpolation.py :: blend_correct()` fed **ground-truth
anchors**. Precisely, per ear:

```
1. raw_full = compute_raw_full(template, mesh.vertices, method="tps")
      -> registration's dense 85-point prediction (error ~3.9mm)

2. gt_anchors = {a: ground_truth[a] for a in ANCHOR_INDICES}
      -> the 15 TRUE anchor positions.  NO MODEL IS USED.

3. blended = blend_correct(raw_full, gt_anchors, mesh.vertices)
```

and `blend_correct` itself does:

```
for each segment between two consecutive anchors:
    corr_start = gt_anchor[start] - raw_full[start]      # correction at each end
    corr_end   = gt_anchor[end]   - raw_full[end]
    for each point in the segment:
        frac = local_index / (n_points_in_segment - 1)   # linear by arc position
        out[i] = raw_full[i] + (1-frac)*corr_start + frac*corr_end

then: snap any point within DEFAULT_SNAP_RADIUS (3.0mm) of the mesh
      onto its nearest vertex
```

So blend = **the registration's own dense shape, slid so it passes exactly
through the true anchors, then lightly snapped to the surface.**

**Scoring:** only the 70 NON-ANCHOR indices
(`np.setdiff1d(arange(85), ANCHOR_INDICES)`). The 15 anchors are exact by
construction here (error 0) and are excluded — including them would drag every
contour mean down and produce different numbers.

Per-contour means count only that contour's non-anchor points: outer_helix 21
(not 25), concha 24 (not 30), inner_helix 17 (not 20), superior_antihelix 8
(not 10).

---

## Exact data split

```
ids        = ds.subject_ids[:60]              # first 60 subjects, sorted
train, test = next(iter(k_fold_subject_split(ids, 2, seed=0)))
```

- **30 train subjects** -> registration template (k_references=7) AND the PCA prior
- **30 test subjects** -> scored, both ears each = **60 ears**
- Template and prior never see a test subject.

---

## Expected output — verify line by line

```
template+prior from 30 subjects; scoring 30
prior: 60 shapes, 45 modes

ears: 60   scoring the 70 NON-ANCHOR points only

        method      mean    median       p95
--------------------------------------------
         blend   1.3457m   1.3808m   1.8122m
         prior   1.5405m   1.4657m   2.2188m
    prior_snap   1.2893m   1.2264m   1.9509m

prior vs blend: +0.1948mm +/- 0.0498 (SE)   95% CI [+0.0972, +0.2925]
  ears improved 19/60   SIGNIFICANT

prior_snap vs blend: -0.0564mm +/- 0.0475 (SE)   95% CI [-0.1495, +0.0367]
  ears improved 33/60   not significant

               contour        blend        prior   prior_snap
-------------------------------------------------------------
           outer_helix      2.1213m      1.9790m      1.6393m
        concha_outline      1.0455m      1.3505m      1.1292m
           inner_helix      1.1073m      1.4565m      1.2230m
    superior_antihelix      0.7169m      1.1379m      0.9917m
```

---

## If your BLEND numbers don't match

| symptom | cause |
|---|---|
| blend ≈ 1.6-2.1mm everywhere | using **predicted** anchors instead of ground truth — this is the #1 cause |
| `ears:` is not 60 | wrong `--n-subjects`, or a different split seed |
| blend mean well below 1.35 | scoring **all 85** points (anchors are exact here, so they pull the mean down) |
| all four contours shifted uniformly | different `--k-references` (must be 7) — this changes the registration and therefore `raw_full` |
| outer_helix fine, others off | scoring full contour ranges instead of non-anchor subsets |
| everything slightly off | different subject ordering — `ds.subject_ids` is `sorted(mesh_dir.glob("*.ply"))`; ensure the mesh folder contains exactly the 200 challenge `.ply` files and nothing else |

**blend depends on the registration template**, so `--k-references` and the
train/test split must match exactly. It does NOT depend on any trained network,
GPU, or torch seed — blend is fully deterministic given the split.

`prior` and `prior_snap` additionally depend on `N_MODES=45` and `RIDGE=0.05`.

---

## Caveats to carry forward with these numbers

1. **Ceilings, not achievable accuracy.** Perfect anchors are assumed. The real
   pipeline's non-anchor error is 1.6023mm, not 1.3457mm.
2. **60 ears only.** The paired SEs are given above; treat differences smaller
   than ~0.1mm as unresolved.
3. **Choosing outer_helix as the prior's contour on the same 60 ears that
   measured it is mild overfitting.** The −0.48mm margin is probably real but is
   not validated on held-out subjects.
4. **The advantage may not survive real anchor error.** The prior degrades +134%
   under realistic anchor noise; blending degrades +19%. See
   `SOLUTION_README.md`, "Why prior interpolation may not survive real anchor
   error".
