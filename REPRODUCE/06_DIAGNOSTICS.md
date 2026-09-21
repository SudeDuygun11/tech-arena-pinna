# Diagnostic scripts — commands and expected output

None of these train a network unless noted, so all are fast and fully
deterministic. **Most use GROUND-TRUTH landmarks — see GOTCHA 1.**

Set `DATA="2026 Munich Tech Arena - Datas"`.

---

## annotation_quantization.py — are labels vertex-snapped?

```bash
python scripts/annotation_quantization.py --data-dir "$DATA" --n-subjects 30
```
Mesh-free-ish, ~2 min. Expect:
```
mean mesh edge length: 0.808mm
GT -> nearest VERTEX : 0.2483mm     NULL (random pts) : 0.2561mm
ratio 0.969   exactly on a vertex: 0.00%
GT -> nearest TRIANGLE SURFACE: 0.0132mm
```
Ratio ~1.0 => labels placed freely on the surface, NOT vertex-quantized.
Vertex spacing is therefore not a floor on achievable accuracy.

---

## shape_prior_capacity.py — is a PCA prior sharp enough?  (D1)

```bash
python scripts/shape_prior_capacity.py --data-dir "$DATA" --n-subjects 200 --folds 5
```
Landmarks only, ~3 min. Expect (85-point model, cross-validated reconstruction
of held-out TRUE configurations):
```
 30 modes  95.5%  0.561mm
 45 modes  98.2%  0.377mm
```
15-anchor model at 45 modes gives 0.000mm — that is FULL RANK and constrains
nothing; ignore it.

---

## interpolation_ceiling.py — mesh-free, prior vs #points observed

```bash
python scripts/interpolation_ceiling.py --data-dir "$DATA" --n-subjects 200 --folds 5
```
~3 min. Expect (all 85 points):
```
PERFECT observations:  15 obs -> 0.640    85 obs -> 0.377
+1.6mm noise:          15 obs -> 1.497    85 obs -> 0.710
```
Shows the prior's noise sensitivity: +134% degradation from perfect to +1.6mm.
Note the noise model here is an independent Gaussian, which
`error_correlation.py` later showed to be UNREALISTIC — do not rely on the
absolute projections.

---

## interpolation_compare.py — blend vs prior vs prior_snap

See `05_INTERPOLATION_TABLE.md` — full detail there.

---

## surface_recall.py — score a preprocessing filter

```bash
python scripts/surface_recall.py --data-dir "$DATA" --n-subjects 12 \
    --filters "all,crest80,crest90,crest95,concave25"
```
~10 min. Expect:
```
all         100.0% faces   99.56% recall<=0.25mm   p95 0.045mm
crest80      19.4% faces   38.82% recall<=0.25mm   p95 3.085mm
```
Measures distance to the nearest retained TRIANGLE, not vertex — vertex
distance overstates and flatters a filter.

---

## candidate_diagnostic.py — registration seed error and candidate recall

```bash
python scripts/candidate_diagnostic.py --data-dir "$DATA" --n-subjects 24 --budget 128
```
~15 min. Expect:
```
seed error: mean 3.910mm  median 3.644  p95 7.714  max 12.074
radius 10mm: 1397 vertices, 0.56% MISS, 97.50% recall<=0.50mm
subsampled to 128: recall collapses to 23.89%
```

---

## candidate_selection_diagnostic.py — does informed selection beat random?

```bash
python scripts/candidate_selection_diagnostic.py --data-dir "$DATA" \
    --n-subjects 24 --radius 10.0 --budgets 64,128,256
```
~20 min. Expect at budget 256: fps 0.542mm mean, random 0.584,
curvature_weighted 0.626, curvature_top 1.094. Curvature-based selection does
NOT beat random.

---

## ranking_feasibility.py — can a ranker reach the candidate ceiling?

```bash
python scripts/ranking_feasibility.py --data-dir "$DATA" --n-subjects 24 --budget 256
```
**Trains a model** (~40 min). Expect:
```
ORACLE 0.534mm   proxy ranker 5.524mm   consensus 3.303mm   random 7.577mm
Spearman(|correction|, error) = +0.301, 93.9% of anchors positive
```

---

## confidence_diagnostic.py — E-CPV vs |correction|  (D2)

```bash
python scripts/confidence_diagnostic.py --data-dir "$DATA" --n-subjects 24 --n-seeds 5
```
**Trains an ensemble** (~15 min). Expect Spearman: E-CPV +0.310 (monotonic
quintiles, positive on all 4 contours), CORR +0.212 (non-monotonic, fails on 2).

---

## prior_strength_diagnostic.py — prior strength sweep  (D3)

```bash
# weak local model
python scripts/prior_strength_diagnostic.py --data-dir "$DATA" --n-subjects 24 --n-seeds 5
# strong local model (adds polish)
python scripts/prior_strength_diagnostic.py --data-dir "$DATA" --n-subjects 100 \
    --n-seeds 5 --inner-folds 4 --with-polish
```
**Trains models** (~15 min / ~45 min). Key result: the optimum moves from
strength 2.0 (−12.7%) at 3.080mm local error to 0.05 (−2.6%) at 1.759mm.
With UNIFORM weights the prior never helps at strong local accuracy.

---

## correlated_noise_test.py — how correlated error defeats a prior

```bash
python scripts/correlated_noise_test.py --data-dir "$DATA" --n-subjects 200 --folds 5
```
Landmarks only, ~5 min. Expect (input 1.48mm, 45 modes): independent 0.706,
5mm 1.116, 10mm 1.130, fully coherent 0.377. Non-monotonic — both extremes are
fine, 5-10mm is the killer.

---

## error_correlation.py — REAL error correlation (needs saved predictions)

```bash
# step 1: produce predictions (~2.5 h, trains models)
python scripts/cross_validate.py --data-dir "$DATA" \
    --n-subjects 100 --outer-folds 2 --inner-folds 4 --k-references 7 \
    --epochs 150 --with-correction --with-polish \
    --capacity large --multi-seed-ensemble --n-seeds 5 \
    --save-predictions preds_baseline.npz --torch-seed 0

# step 2: analyse (instant)
python scripts/error_correlation.py --predictions preds_baseline.npz
```
Expect:
```
raw error 1.7834mm
rigid component 0.1715mm (9.6%)
0-2mm +0.776 | 4-6mm +0.378 | 8-12mm +0.065 | 12-45mm ~0 to -0.05
fitted correlation length L ~ 4.5mm  -> WORST BAND
```
This explains the shape prior's null result and shows the synthetic Gaussian
model overestimates short-range correlation by up to 0.2.

---

## crest_paired_test.py — the paired-comparison template

```bash
python scripts/crest_paired_test.py --data-dir "$DATA" --n-subjects 48 \
    --n-seeds 3 --inner-folds 4
```
**Trains an ensemble** (~20 min). Expect:
```
crest OFF 2.4077   crest ON 2.4082
difference +0.0005mm +/- 0.0057 (SE)   95% CI [-0.011, +0.012]
control: anchors diff 0.000000mm       superior_antihelix diff 0.0000mm
```
Copy this script's structure for any post-processing comparison — the exact
0.000000mm controls are what make the null trustworthy.
