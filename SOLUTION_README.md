# Pinna Landmark Extraction — Solution Documentation

Solution for the ATL/Huawei Tech Arena 2026 challenge: extract 85 ordered 3D
landmarks per ear (outer helix 25pts, concha outline 30pts, inner helix 20pts,
superior antihelix 10pts) from a 3D head mesh, scored by mean Euclidean
landmark distance.

## Purpose

Individual pinna shape drives a person's head-related transfer function (HRTF),
the acoustic filter that lets someone localize sound in 3D. Measuring HRTFs
acoustically per person doesn't scale, so extracting anthropometric pinna
landmarks from a 3D scan is a practical step toward cheap individualized
spatial audio.

## Architecture

Two-stage hybrid, chosen because only 15 of the 85 points are true anatomical
anchors judged independently by a human; the other 70 are algorithmically
redistributed along contours between anchors (validated across all 200
subjects, and confirmed by Huawei on the challenge platform).

```
mesh -> registration (1a) -> anchor correction (1b)
     -> blend interpolation (2) -> polish network (2b) -> 85 landmarks
```

- **1a Registration** (`src/registration.py`): reference subjects chosen by
  Generalized Procrustes medoid selection are non-rigidly warped onto the
  target (rigid ICP + thin-plate spline), giving a dense 85-point prior. TPS
  is the validated default.
- **1b Anchor correction** (`src/anchor_model.py`): a PointNet-style network
  with a per-anchor embedding predicts a 3D correction for each of the 15
  anchors from a local surface patch.
- **2 Blend interpolation** (`src/interpolation.py`): the other 70 points are
  produced by blending corrected-anchor deltas into the registration's own
  dense shape, plus a light surface snap.
- **2b Polish**: a second network refines all 85 points on top of Stage 2.
- Both ears share one model via canonical mirroring (`src/canonical.py`).

## Error reduction journey

| Stage | mean error (mm) | n_ears | Change |
|---|---:|---:|---:|
| Registration-only baseline | 3.86 | — | — |
| + Learned anchor correction | 2.87 | — | −26% |
| + Polish network (100 subjects) | 1.98 | — | −31% |
| + Full 200-subject scale | 1.7503 | 400 | −12% |
| + More epochs & snapshot ensembling | 1.7080 | 400 | −2% |
| + Large capacity & multi-seed ensemble | **1.5931** | **400** | −7% |

**Best validated result: 1.5931mm** — `--capacity large --multi-seed-ensemble
--n-seeds 3`, full 3-fold subject-wise CV, leakage-free.

> A previously-reported **1.58mm** was withdrawn. It came from a
> `--train-fraction 0.90` split scored on **40 ears**, not the 400-ear CV every
> other row uses, and changed three things at once (capacity, contour
> weighting, AND protocol). The 90/10 split trains on more subjects and tests
> on 10x fewer, so it is biased low and not comparable.

**Never validated at full CV scale** (each has isolated-test support only):
`--capacity xlarge`, `--contour-weighted-loss`, `--n-seeds 5`.

---

## ⚠ Experimental methodology — read before trusting any comparison

### 1. Training is nondeterministic unless seeded

`torch.manual_seed` was never called in this codebase; `--seed` only controlled
numpy (splits, augmentation). Every run therefore trained a differently
initialised network. Running the SAME config twice (24 subjects, xlarge, 150
epochs) gives:

| Metric | run A | run B | noise |
|---|---:|---:|---:|
| overall | 2.9970 | 3.1112 | **0.114** |
| anchors only | 2.9983 | 3.1168 | 0.119 |
| inner_helix | 3.8952 | 3.7435 | 0.152 |
| superior_antihelix | 3.1124 | 3.9663 | **0.854** |

**The noise floor (~0.11mm overall) exceeds most effects worth measuring.**
A `--torch-seed` flag now exists.

### 2. Pair whatever can be paired

If a change happens AFTER training, both arms can share one set of trained
models and be evaluated on the same ears — training variance then cancels
exactly. This is worth roughly a 10x tighter bound for the same compute:

| crest attraction test | difference | uncertainty | conclusion |
|---|---:|---:|---|
| unpaired (two training runs) | −0.012mm | ±0.114 | "unresolved" |
| **paired (shared models)** | **+0.0005mm** | **±0.011** | **definitive null** |

Pairable: post-processing, inference-time changes, selection rules, shape-prior
fitting, TTA, crest attraction. NOT pairable: capacity, pooling, input
features, loss — these change training itself.

### 3. Prediction track record — weight optimism accordingly

Over one extended session, **every optimistic "this should help" prediction was
wrong; every pessimistic structural prediction held.**

Wrong: Gaussian curvature would help (null) · a better seed would fix candidate
localization (refuted) · confidence weighting negligible (reversed twice, ended
at nothing) · shape prior worth 1-2% at full scale (0.5%, n.s.) · 400-ear CI
would be +/-0.004mm (was +/-0.032) · fewer prior modes would reject correlated
error (refuted) · prior would beat blending on paper (significantly worse) ·
geodesic patches would help (+35% worse, though confounded).

Held: prior gain would shrink as the local model improved (decayed to zero) ·
pairing would tighten bounds (10x) · oracles are not achievable targets ·
component-level wins mostly fail to survive integration.

**Observed translation rate:** three isolated wins failed end-to-end (CPD
registration −3-4% isolated but +4.5% worse in-pipeline; crest attraction's
geometric premise measured 2.3x better than chance but produced exactly zero;
shape prior −2.6% at 100 subjects, −0.5% n.s. at 400). Discount component-level
results to roughly 30% when projecting end-to-end.

### 4. For unpairable changes, budget seeds honestly

With noise sd ~0.08mm: detecting 0.15mm needs ~2 seeds/arm, 0.10mm needs ~4,
0.05mm needs ~10. A 3% effect at 24-subject scale needs ~20 runs to establish.
Prefer larger test sets — noise falls as 1/sqrt(n), so 400 ears is ~2.9x
quieter than 48.

---

## What worked (validated)

- **Network capacity** — the largest single lever historically; ~22K to ~348K
  params gave −19.3% at isolated scale. Now believed exhausted at this error
  level (a teammate's controls at ~1.55mm showed no capacity gains).
- **Multi-seed ensembling** — −5.5% standalone, stacks with capacity.
- **More training data** — 100→200 subjects gave −11.6%; never saturated.
- **The blend-interpolation redesign.** An early version traced independent
  mesh-surface shortest paths between anchors; it broke on `outer_helix`
  (~3.7mm even with perfect anchors) because the shortest path cuts across the
  sulcus behind the ear. Blending into the registration's own dense shape
  fixed it uniformly.
(The shape prior was moved OUT of this list — it measured as no effect at full
scale. See the shape-prior section.)

## What was tried and rejected

- **Hard-example mining** (`--hard-example-mining`, worst 25% boosted x2, one
  extra training round): **+3.7% worse** -- 1.8489mm vs 1.7834mm same-config
  baseline (200 ears). Only inner_helix improved (-3.2%); outer_helix was hurt
  badly (+11.4%).
  The motivating argument was an inversion: REMOVING the noisiest 15% of
  examples had cost +7.1%, so those examples appeared to carry signal, so
  emphasising them should help. **Both halves cannot be true, and the
  resolution is that the two operations are not symmetric**: removing data
  starves an underfit model, while over-weighting a subset starves everything
  else. Hard examples are worth KEEPING but not worth OVER-WEIGHTING.
  Note also the training error at mining time was 0.495-0.590mm mean vs ~1.7mm
  test error, so the model fits its training set considerably better than
  held-out data -- part of the residual training error is likely label noise
  rather than difficulty, which further undermines boosting it.
- **Wing loss**: +7.1% worse at default hyperparameters.
- **`xxlarge` capacity** (~64x params): diminishing returns vs `xlarge` (~16x).
  Capacity is considered exhausted -- a teammate's controls at ~1.55mm also
  showed no capacity gains (RBF 1.5806, bilateral-feature 1.5924, reduced-rank
  1.5785 vs plain linear 1.5753).
- **Jitter recalibration**: INCREASING the augmentation jitter above the 1.5mm
  default made results worse, so the original value is already near-optimal.
  Note this is jitter on the patch CENTRE (simulating registration error), not
  noise on the patch points -- the latter is untested.
- **Boosting landmark 74's loss weight** (`--anchor74-boost`): rejected. Index 74
  is the inner_helix "derived continuation" endpoint, structurally different
  from the other 14 true anatomical anchors, and was independently flagged as
  worth special treatment by an external comparison of 58 solutions. Extra loss
  emphasis did not help.
- **Snapshot ensembling** (cyclic LR, one snapshot per cycle): works, but
  multi-seed ensembling is more diverse and was preferred; the 1.7080mm entry in
  the error table used snapshots.
- **Per-contour specialization** (4 independent networks): +11.5% worse —
  fragmenting a small training set hurt more than specialization helped.
  NOTE: a teammate's shared-trunk / separate-head variant (S54) is their
  strongest standalone method, so the *shared representation* is the key
  difference. Untested by us.
- **PCA/ASM as a registration prior**: unstable — traced to the fitting
  procedure (hard NN correspondence, per-axis coefficient clipping), not the
  model family. See the shape-prior section: PCA works well when fitted properly.
- **CPD / hybrid registration**: ~3-4% better at the registration level, +4.5%
  worse under the correction network, ~25x slower.
- **Symmetric / inverse-consistent registration**: round-trip drift was only
  3.7% of actual error — little asymmetry to correct.
- **Increased weight decay**: +6.4% worse (the model is underfit, not overfit).
- **Test-time augmentation** (patch resampling): no effect.
- **Training-example filtering by ensemble disagreement**: +7.1% worse.
- **Crest attraction** (`--crest-attract`): **definitive null** —
  +0.0005mm ± 0.0057 SE, 95% CI [−0.011, +0.012], 25/48 ears improved, both
  controls exact (anchors 0.000000mm; excluded contour 0.0000mm). Explained by
  the filter measurement below: a plain curvature threshold reaches only 38.8%
  recall@0.25mm, so the attraction target is usually the wrong surface.
- **Geodesic patches** (`--geodesic-patch`): build the patch from an
  along-surface neighbourhood instead of a Euclidean ball. Motivated by a real,
  measured defect -- the helix folds over itself, and **33.5% of Euclidean-ball
  vertices are across a fold** (54.6% for superior_antihelix), i.e.
  Euclidean-near but surface-far.
  Result: fold 1 gave **2.4356mm vs 1.8008mm** for the same-config Euclidean
  baseline (+35%). **But the test was confounded and should not be read as a
  verdict on geodesic neighbourhoods:**
    - patch vertex count fell 928 -> 582, so the receptive field SHRANK; this
      changed two things at once (neighbourhood shape AND size)
    - zero-padding rose from 3% to 17% of patches (40% for superior_antihelix)
      because `n_points` stayed at 256 while the neighbourhood shrank, injecting
      fake origin points with zero normals
  A fair re-test needs a LARGER geodesic radius (~9-10mm) chosen to match the
  Euclidean vertex count, so only neighbourhood shape varies.
  Also worth questioning the premise: across-fold geometry may be SIGNAL
  ("you are next to a fold") rather than contamination.
- **Interpolation by shape prior instead of blending** (`interpolation_compare.py`,
  60 ears, perfect anchors, scoring the 70 non-anchor points):

  | method | mean | vs blend |
  |---|---:|---|
  | blend (current) | 1.3457mm | -- |
  | prior alone | 1.5405mm | **+0.195 +/- 0.050, significantly WORSE** |
  | prior + surface snap | 1.2893mm | −0.056 +/- 0.048, not significant |

  The prior gets shape roughly right but places points off the mesh surface --
  the snap recovers 0.25mm. Per contour it wins decisively on **outer_helix**
  (1.6393 vs 2.1213, the longest contour where blending drifts most) and loses
  on the other three. A per-contour hybrid computes to 1.201mm vs blend's
  1.346mm (−10.8%), but that selects the winner on the same data that measured
  it and needs held-out validation. Given interpolation only costs 0.042mm
  end-to-end (see error structure above), the achievable gain is ~1-3%.
- **Attention pooling / multiscale patches / Gaussian curvature**: all measured
  as single runs against a mis-specified baseline; re-scored against a
  same-config baseline (~3.054mm) they are +0.16, −0.16 and +0.03mm — at or
  within the 0.114mm noise floor. **Unresolved rather than rejected**;
  multiscale is the most promising and deserves a multi-seed re-test.

---

## Preprocessing / mesh-filter findings (surface recall)

`scripts/surface_recall.py` scores a filter by the share of GT landmarks within
X mm of a **retained triangle** — i.e. did preprocessing throw the answer away?

| filter | faces kept | recall ≤0.25mm | p95 | worst |
|---|---:|---:|---:|---:|
| all (no filter) | 100% | 99.56% | 0.045mm | 1.980mm |
| crest80 (plain curvature threshold) | 19.4% | **38.82%** | 3.085mm | 5.708mm |
| corridor 70/60/4 | 61.0% | 83.43% | 1.449mm | 5.847mm |
| corridor 60/40/4 | 78.0% | 93.87% | 0.356mm | 2.810mm |
| *(teammate's prediction-seeded corridor)* | *28.7%* | *99.6%* | *0.039mm* | *3.622mm* |

- **Measure to the triangle, not the vertex** — landmarks sit mid-triangle, so
  vertex distance overstates and flatters a filter. `geometry.nearest_surface_points()`.
- **A plain curvature percentile is a bad filter** — scattered specks, not a
  continuous ridge.
- **Connectivity + dilation helps but trades badly** — ~78% of faces must be
  kept for ~94% recall.
- **Connectivity adds a new failure mode**: discarding small components can
  delete every fragment near a landmark, giving *worse* worst-case (13.2mm).
- **Design tension**: high recall, low faces-kept, non-prediction-reliance —
  you can have two, not all three. Prediction-seeded filters achieve the first
  two but inherit the coarse model's mistakes as permanent losses (teammate's
  worst case: 6.08mm, i.e. the true landmark deleted outright).

---

## Shape prior (local ↔ global joint fitting)

`src/joint_fit.py` (skeleton), `scripts/shape_prior_capacity.py`,
`scripts/confidence_diagnostic.py`, `scripts/prior_strength_diagnostic.py`.

Motivation: the 15 anchors are currently corrected **independently**, with
nothing enforcing that they form a plausible ear.

**D1 — prior family and size.** PCA/PDM on all 85 points. Reconstruction of
held-out TRUE configurations, cross-validated:

| model | modes | var | recon error | constrains? |
|---|---:|---:|---:|---|
| 15-anchor (45 dims) | 20 | 94.0% | 0.679mm | weakly (20/45) |
| 15-anchor | 45 | 100% | 0.000mm | **no — full rank** |
| **85-point (255 dims)** | **45** | **98.2%** | **0.377mm** | **strongly (45/255)** |

The 85-point model is sharper AND constrains more: the 70 interpolated points
are extra observations that regularize the fit. Mode count chosen by CV, not a
variance threshold (arXiv:1808.00309 shows the 90-98% heuristic is unprincipled).

**D2 — confidence.** Per-anchor confidence via **E-CPV** (ensemble coordinate
prediction variance; the strong baseline in arXiv:2203.02351 — their other two
measures are heatmap-derived and don't apply since we regress coordinates).
Spearman(E-CPV, error) = +0.310, monotonic quintiles, positive on all four
contours. |predicted correction| is worse (+0.212, non-monotonic, fails on two
contours) — discarded.

**D3 — prior strength.** Depends critically on local model quality:

| local model | local-only | best gain | optimal strength |
|---|---:|---:|---:|
| weak (12 subj, no polish) | 3.080mm | −12.7% | 2.0 |
| **strong (50 subj + polish)** | **1.759mm** | **−2.6%** | **0.05** |

**With uniform weights the prior NEVER helps at strong local accuracy** (every
strength is worse than no prior). It only works with E-CPV weighting: 1.714mm
vs 1.759mm baseline, versus 1.774mm unweighted.

Why confidence matters more as the model improves — the calibration intercept
(error E-CPV cannot see) collapses from 2.108mm to 0.361mm.

**FINAL VERDICT (400-ear full CV, paired): NO EFFECT. Do not use.**

| | baseline | +prior |
|---|---:|---:|
| overall | 1.5948 | 1.5876 |
| anchors only | 1.5600 | 1.5351 |

Paired effect **−0.0072mm +/- 0.0164 SE**, 95% CI [−0.039, +0.025], 196/400
ears improved (a coin flip). The baseline arm reproduced the 1.5931mm
reference to 0.002mm, so this is not a drift artefact.

The gain decays to nothing as the local model improves -- the whole point of a
prior is removing noise, and there is progressively less to remove:

| local model error | prior gain |
|---:|---:|
| 3.080mm (12 subjects) | −12.7% |
| 1.759mm (50 subjects) | −2.6% |
| **1.595mm (400 ears)** | **−0.5%, not significant** |

One real signal inside the null: anchors improved −0.025mm while outer_helix
(+0.026) and concha_outline (+0.008) got WORSE. The prior does help the 15
points it constrains; that gain then fails to propagate through blend
interpolation to the other 70 -- further evidence that interpolation, not
anchor accuracy, is the binding constraint.

**Methodology note:** the 400-ear CI came out at +/-0.032mm, 8x wider than a
sqrt(n) projection from the 48-ear crest test predicted. Per-ear variance of
the effect -- not sample size -- dominates here. Do not project one
experiment's confidence interval from another's variance.

---

## Error structure: what actually limits us

Measured on the 400-ear run, this is the decomposition that should drive
priorities:

| | error |
|---|---:|
| anchors (15 points, **predicted**) | **1.5600mm** |
| non-anchor (70 points, **interpolated**) | **1.6023mm** |
| overall | 1.5948mm |

**Interpolation costs only 0.042mm.** Predicting all 85 points directly would
land near anchor quality (~1.56mm) -- a 2.6% gain, not a route to sub-1mm.

An earlier version of this document claimed interpolation was the binding
constraint, reasoning from "blend_correct with PERFECT anchors gives 1.35mm, so
70x1.35/85 ~ 1.1mm is the floor". That conditions on an anchor model with zero
error, which is neither available nor approachable. Against what we can
actually achieve per point (1.56mm), interpolation is nearly free.

**The real constraint is per-point prediction accuracy (~1.56mm).** Reaching
0.9mm requires a 42% better per-point predictor -- the thing that has resisted
capacity, ensembling, attention, multiscale features, Gaussian curvature, crest
attraction, candidate ranking, and shape priors.

### Error correlation (measured, 200 ears)

From `scripts/error_correlation.py` on saved predictions:

| separation | correlation |
|---|---:|
| 0-2mm | +0.776 |
| 2-4mm | +0.666 |
| 4-6mm | +0.378 |
| 8-12mm | +0.065 |
| 12-45mm | ~0 to −0.05 |

**Fitted correlation length L ~ 4.5mm** -- the regime where a shape prior
filters worst. This EXPLAINS the prior's null result rather than leaving it
unexplained. Only **9.6%** of error is a global rigid shift (the part Procrustes
removes for free); the rest is local deformation.

The synthetic Gaussian-random-field model in `correlated_noise_test.py` does
NOT match reality: it overestimates short-range correlation by up to 0.2 and
cannot represent the negative correlation seen at 16-45mm (opposite sides of
the ear err in opposite directions). **Its projected numbers should not be
relied on.**

## Key structural findings

- **Interpolation is deterministic**: non-anchor landmarks are evenly spaced by
  chord length (CV 0.5-1.8%) and sit within 0.2-0.3 units of the surface. Only
  the 15 anchors need learning.
- **Labels are NOT vertex-quantized.** GT-to-nearest-vertex is 0.2483mm vs a
  null of 0.2561mm for random points on the same triangles (ratio 0.969); 0.00%
  sit exactly on a vertex; GT-to-surface is 0.0132mm. Annotators placed points
  continuously, so vertex spacing (0.808mm) is not a label floor.
  *Annotator judgment noise remains unmeasured — no duplicate annotations exist.*
- **Landmarks track curvature ridges on 3 of 4 contours**: outer_helix 0.81mm,
  inner_helix 0.98mm, concha_outline 1.08mm vs 2.63mm for random points (ratio
  0.43); superior_antihelix 2.28mm (ratio 0.87, no signal). **Caveat**: measured
  from truth outward, so it says the ridge passes near the answer, NOT that
  moving a prediction toward the ridge lands on it — acting on it did nothing.
- **Contour length predicts error better than ridge detectability.**
  superior_antihelix has no ridge signal yet is the second-best contour
  (1.54mm) because it is short and tightly pinned between two anchors; the long
  contours drift more (outer_helix 1.84mm/25pts, inner_helix 1.97mm/20pts).
- **The pipeline is underfit, not overfit**, at every scale tested.
- **Oracles are not targets.** Candidate-set "oracle" error scales with
  candidate density (0.534mm at 256 candidates, 0.211mm at ~1400, → 0 for
  continuous surface points). A teammate's 0.881mm anchor oracle yielded a
  realized 0.038mm improvement.

## Candidate-ranking approach (evaluated, not built)

`src/candidates.py` (skeleton). Registration seed error is 3.910mm mean /
7.714mm p95, so a 10mm ball is needed for <1% miss — holding ~1400 vertices.
Subsampling to 128 collapses recall@0.5mm from 97.5% to 28.3%, and **no
selection strategy fixes it**: FPS 0.542mm mean, random 0.584, curvature-weighted
0.626, curvature-top 1.094 (worst). Better seeds barely help — post-polish seed
improves the mean 27% but p95 only 12%.

Feasibility without training a ranker: proxy ranker **5.524mm** vs 0.534mm
oracle, barely better than random (7.577mm); consensus voting 3.303mm, worse
than the existing pipeline. Separability Spearman +0.301 (93.9% of anchors
positive) — signal exists but is weak.

**Not recommended.** Patches ~1.2mm apart are near-identical, and the ranker
must distinguish them.

## Repository layout

```
src/
  contours.py        -- landmark/contour/anchor index definitions
  geometry.py        -- Kabsch, mirroring, nearest_surface_points (exact
                        point-to-TRIANGLE distance)
  dataset.py         -- mesh/landmark CSV loading
  canonical.py       -- left/right-mirrored canonical-frame access
  template.py        -- reference-ensemble construction (GPA medoid selection)
  registration.py    -- TPS / CPD / hybrid non-rigid registration
  shape_model.py     -- early PCA/ASM prior (superseded; see joint_fit)
  patch_features.py  -- local patch features; crop_submesh
  anchor_model.py    -- PatchCorrector, training loops, ensembling
  interpolation.py   -- Stage-2 blend interpolation + crest_attract (null)
  ridge_filter.py    -- connected ridge-corridor filtering
  joint_fit.py       -- local<->global iterative fitting (SKELETON)
  candidates.py      -- candidate generation + ranking (SKELETON, not recommended)
  training_data.py   -- leakage-free example generation
  splits.py          -- subject-wise k-fold / single-split
  pipeline.py        -- full inference (predict_canonical)
  estimator.py       -- challenge entry point (LandmarkExtractor)
  metrics.py         -- official metric

scripts/
  cross_validate.py              -- main CV harness
  build_template.py              -- build & save a registration Template
  train_anchor_model.py          -- train & save a Stage-1b correction network
  surface_recall.py              -- score a preprocessing filter
  candidate_diagnostic.py        -- seed error, vertex counts, candidate recall
  candidate_diagnostic_cascade.py-- the same for three seed sources
  candidate_selection_diagnostic.py -- selection strategies at fixed budget
  ranking_feasibility.py         -- can a ranker reach the ceiling?
  annotation_quantization.py     -- are labels vertex-snapped?
  shape_prior_capacity.py        -- D1: is the prior sharp enough?
  confidence_diagnostic.py       -- D2: E-CPV vs |correction|
  prior_strength_diagnostic.py   -- D3: prior strength sweep
  crest_paired_test.py           -- paired-comparison template
  interpolation_ceiling.py       -- mesh-free: prior interpolation vs #points observed
  interpolation_compare.py       -- blend vs prior vs prior+snap, given GT anchors
  correlated_noise_test.py       -- how correlated error defeats a shape prior
  error_correlation.py           -- measures REAL error correlation from saved preds
```

## Running the CV harness

Best **validated** configuration (1.5931mm, 400 ears):

```bash
python scripts/cross_validate.py --data-dir "<mesh/landmarks parent>" \
    --n-subjects 200 --outer-folds 3 --inner-folds 4 --k-references 7 \
    --epochs 150 --with-correction --with-polish \
    --capacity large --multi-seed-ensemble --n-seeds 3
```

Best **proposed** configuration — every ingredient has isolated-test support,
but this exact command has never completed end to end:

```bash
python scripts/cross_validate.py --data-dir "<mesh/landmarks parent>" \
    --n-subjects 200 --outer-folds 3 --inner-folds 4 --k-references 7 \
    --epochs 150 --with-correction --with-polish \
    --capacity xlarge --multi-seed-ensemble --n-seeds 5 \
    --contour-weighted-loss --torch-seed 0
```

**Memory note:** full-scale runs previously died with `MemoryError` during mesh
loading. `generate_examples` / `generate_polish_examples` now free the full
trimesh object (and its cached edge/normal arrays) after cropping. Do not run
two mesh-loading jobs concurrently.

## What error is realistically reachable

| step | expected | basis |
|---|---:|---|
| current validated | **1.5931mm** | measured twice (1.5931, 1.5948) |
| + per-contour interpolation hybrid | ~1.56 | −11% on a ceiling worth 0.042mm end-to-end |
| + `xlarge` / 5 seeds / contour-weighting | ~1.52-1.56 | isolated tests, never validated at scale |
| + bilateral | −1-2% | teammate's repeated result |
| **realistic landing** | **~1.50-1.55mm** | |
| best case if all stack | ~1.45mm | |

**0.9mm is not reachable by anything measured here.** It needs a per-point
predictor ~42% better than 1.56mm. The only lever with a proven rate is more
labelled data (−11.6% per doubling of 100->200), which would need roughly five
doublings.

Two things could change that assessment, and both are cheap:
1. **Ask the organizers whether repeat annotations exist.** Annotator judgment
   noise is unmeasured and could be ~1mm, in which case 0.9mm sits below the
   labels' own precision. (Vertex quantization is already ruled out -- see
   structural findings.)
2. **Establish what any reported sub-1mm result was measured on.** Three times
   in this project, numbers proved non-comparable across protocols -- twice they
   were our own (the withdrawn 1.58mm; a teammate's 1.5126mm on a reused
   population vs 1.7126mm on disjoint folds). A 43% gap over 58 documented
   methods would imply a structurally different approach, which would be worth
   more than every incremental item on this list combined.

## Per-contour specialization: what works and what doesn't

A recurring idea is treating `outer_helix` differently, since it is the worst
contour (1.83mm) and the only one where prior interpolation beat blending.
Several versions have been considered; they are NOT equivalent:

| version | verdict |
|---|---|
| 4 INDEPENDENT per-contour networks | **tested, +11.5% worse** -- each net saw 1/4 of the data, and this pipeline is underfit at every scale |
| 2 independent nets (outer_helix vs rest) | same mechanism; the outer_helix net would train on 1,600 of 6,000 examples (27%) |
| 2 IDENTICALLY-trained nets, split at inference | **pointless** -- identically-trained models are statistically exchangeable, so the assignment is arbitrary. Costs the ensemble average (−5.5%) for no expected gain |
| **shared trunk + separate per-contour heads** | **untried by us; a teammate's S54 is their strongest standalone method (1.5510mm).** Trunk sees all data, only small heads specialize |
| **per-contour INTERPOLATION choice** | untested in-pipeline; see below. Cheapest of all -- it is post-processing, so it needs no retraining and can be measured PAIRED |
| per-contour patch radius | speculative; being tested (see below) |

**The key structural point:** `outer_helix` has only 4 anchors among its 25
points. A separate *prediction* network would improve 4 points and leave 21
untouched. What actually differs for that contour is INTERPOLATION, which is
what the measurement showed.

**Interpolation is not a per-model property.** blend vs prior happens in Stage 2,
after the anchor model. One model can use different interpolation per contour --
no second model required.

### Why prior interpolation may not survive real anchor error

The 1.6393 vs 2.1213 advantage on outer_helix was measured with GROUND-TRUTH
anchors. Two reasons it may not transfer:

1. **Global fit vs local blending.** Blending confines an anchor's error to its
   two adjacent segments. The prior fits a global 45-mode model, so one bad
   anchor shifts the coefficients and moves ALL 85 points. This holds even for
   perfectly independent errors.
2. **Measured noise sensitivity.** Prior reconstruction degrades 0.640 -> 1.497mm
   (+134%) going from perfect anchors to +1.6mm error. Blending on the same
   points degrades 1.3457 -> 1.6023mm (**+19%**) between perfect and real anchors.

Correlated anchor errors (L ~ 4.5mm) compound the first mechanism, which is the
same failure that killed the anchor-level prior. Note that decorrelating the
anchors would NOT fix mechanism 1 -- that is structural.

## External data and pretraining (investigated, not pursued)

- **SONICOM** (~400 unlabelled head scans, Imperial College). Ten watertight
  subjects were downloaded and verified: they load cleanly in trimesh, are in
  millimetres, and match our dataset's scale. **The blocker is ear
  localization** -- SONICOM provides full head+torso scans with NO landmarks, so
  the ear region must be found automatically. Three approaches failed:
  curvature-based seeding, lateral-protrusion heuristics, and a neck-bottleneck
  crop, all defeated by head pose not being axis-aligned. A literature search
  found no published code for this; notably, the one paper that DOES use SONICOM
  3D scans for HRTF work solved it by *having landmarks already*
  (arXiv:2603.24104), and EarNet-style learned ear segmentation
  (IEEE 9622263) has no public release.
- **Self-supervised / masked pretraining**: a teammate measured this directly --
  their S43 (masked geometric pretraining then fine-tuning) scored **2.8813mm**
  vs **2.5522mm** for the same architecture without it. **Measured NEGATIVE**,
  which is the main reason the SONICOM pretraining path was deprioritised rather
  than the localization difficulty alone.
- **Transfer learning from generic 3D/point-cloud or ear-anatomical models**:
  investigated (AudioEar, i3DMM, LmPT). None directly usable -- domain and
  format mismatch.
- **HUTUBS** (96 subjects): landmark format is 12 points, incompatible with the
  85-point specification.

The general finding: more *labelled* data is the one lever with a proven rate
(−11.6% per doubling of 100->200). Unlabelled data has no demonstrated path to
helping here.

## Untried directions for a better per-point predictor

Ranked. Nothing here plausibly delivers 42%, but this is the unexplored space:

1. **Heatmap / distributional prediction** instead of direct coordinate
   regression. Largest untried change and the field's dominant paradigm --
   the landmark-uncertainty literature we drew on is heatmap-based throughout.
   Would also unlock the S-MHA / E-MHA uncertainty measures we had to skip.
2. **Shared trunk + per-contour heads.** Our 4 INDEPENDENT per-contour networks
   were +11.5% worse, but a teammate's shared-trunk variant (S54) is their
   strongest standalone method at 1.5510mm. We tested the failing variant only.
3. **Geodesic patches, done properly** -- larger radius matched to Euclidean
   vertex count (see rejected list for why the first attempt was confounded).
4. **More points per patch** (256 -> 512, `--patch-points`). A 7mm patch holds
   ~928 vertices, so the default discards most of the available geometry.
   Wired and smoke-tested, never run at scale.
   Related: **`--patch-radius`** (default 7.0) is now exposed. A run at 12mm is
   IN PROGRESS against the 7mm baseline below; the motivation (outer_helix has
   the longest arc and most drift, so may want more context) is speculative --
   no measurement supports a per-contour radius, and global multiscale patches
   landed within noise.

   7mm baseline for that comparison (100 subjects, 2 folds, 5 seeds,
   `--torch-seed 0`, with polish) -- **1.7834mm** over 200 ears:

   | contour | 7mm |
   |---|---:|
   | outer_helix | 2.1082 |
   | concha_outline | 1.2809 |
   | inner_helix | 2.1134 |
   | superior_antihelix | 1.8192 |
   | anchors only | 1.7271 |

   **RESULT: 12mm is decisively WORSE -- 2.5606mm vs 1.7834mm (+44%), 200 ears.**
   Clean test (larger patches hold MORE vertices, so no zero-padding artefact,
   unlike the geodesic attempt). 7mm is not arbitrary: the model's accuracy
   lives in fine local geometry, and trading sampling density for coverage
   (256 points now spanning ~3x the area) costs far more than the extra
   context is worth.

   | contour | 7mm | 12mm | change |
   |---|---:|---:|---:|
   | outer_helix | 2.1082 | 2.6720 | **+27%** (least harmed) |
   | inner_helix | 2.1134 | 2.9283 | +39% |
   | superior_antihelix | 1.8192 | 2.8211 | +55% |
   | concha_outline | 1.2809 | 2.1357 | **+67%** (most harmed) |
   | overall | 1.7834 | 2.5606 | +44% |

   **Reusable finding -- context tolerance ranking** (held on both folds):
   `outer_helix -> inner_helix -> superior_antihelix -> concha_outline`.
   outer_helix minds losing local detail least; concha minds it most, which
   fits their anatomy (a long smooth rim vs a tight bowl of fine folds).
   If a global-branch architecture is ever built, this is the prediction it
   must satisfy: outer_helix's head should weight global features highest.
   But there is NO CROSSOVER -- every contour prefers 7mm -- so there is no
   per-contour radius split to exploit.
5. **Iterative refinement** -- currently exactly two passes (correction,
   polish); three or four with patch re-extraction is standard cascade practice.
6. **Local surface-frame prediction** -- (dt1, dt2, dnormal) rather than
   unconstrained XYZ, enforcing the surface constraint measured at 0.0132mm.
7. **Graph/mesh-topology networks** (GCN, FeaStNet) -- uses real edge
   connectivity; larger build, data-hungry.

## Open directions

- **Bilateral / paired-ear constraint** — never built; repeatedly positive for
  a teammate (S48, S51, and inside their best result S58).
- **Shared-trunk + per-contour heads** (teammate's S54 pattern) — our
  *independent* per-contour networks failed, but shared-representation was
  never tried.
- **Multiscale patch features** — best of four unresolved experiments; needs a
  multi-seed re-test.
- **Local surface-frame displacement** — predict (Δtangent₁, Δtangent₂, Δnormal)
  rather than unconstrained XYZ. Landmarks provably lie on the surface
  (0.0132mm), a constraint currently unused.
- **Annotation noise floor** — unmeasurable here; worth asking the organizers
  whether repeat annotations exist. This bounds what any method can achieve.

## Requirements

See `requirements.txt`: `torch`, `trimesh`, `scipy`, `numpy`.
