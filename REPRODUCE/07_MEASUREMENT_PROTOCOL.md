# How to run a comparison that means something

Learned the hard way — four experiments were run against an incomparable
baseline before this was understood.

## 1. Pair whatever can be paired

If the change happens AFTER training, both arms can share one set of trained
models on the same ears. Training variance then cancels exactly.

| crest attraction | difference | uncertainty | conclusion |
|---|---:|---:|---|
| unpaired (2 training runs) | −0.012 | ±0.114 | "unresolved" |
| **paired (shared models)** | **+0.0005** | **±0.011** | **definitive null** |

**10x tighter for the same compute.**

PAIRABLE: post-processing, inference-time changes, selection rules, shape-prior
fitting, TTA, crest attraction, interpolation choice.
NOT PAIRABLE: capacity, pooling, patch radius/points, input features, loss —
these change training itself.

`scripts/crest_paired_test.py` is the template. `cross_validate.py --shape-prior`
shows paired reporting built into the harness.

## 2. Know the noise floor

Same config, twice, 24 subjects:

| metric | A | B | noise |
|---|---:|---:|---:|
| overall | 2.9970 | 3.1112 | **0.114mm** |
| superior_antihelix | 3.1124 | 3.9663 | **0.854mm** |

Contour-level noise is much larger than overall noise. Fold-to-fold spread at
100 subjects was 0.12mm (1.5323 vs 1.6510).

## 3. Budget seeds honestly

With sd ~0.08mm: detecting 0.15mm needs ~2 seeds/arm, 0.10mm ~4, 0.05mm ~10.
A 3% effect at 24-subject scale needs ~20 runs. Prefer larger test sets —
noise falls as 1/sqrt(n).

## 4. Do not project one experiment's CI onto another

A sqrt(n) projection from the 48-ear crest test predicted ±0.004mm at 400 ears.
The actual shape-prior CI was **±0.032mm — 8x wider**. Per-ear variance of the
effect, not sample size, dominated.

## 5. Change one thing

The withdrawn 1.58mm changed capacity, loss weighting AND evaluation protocol
simultaneously. Nothing could be attributed.

## 6. Discount component-level wins ~70% when projecting end-to-end

Observed translation rate:

| change | isolated | in pipeline |
|---|---|---|
| CPD registration | −3-4% | **+4.5% worse** |
| crest attraction | ridges 2.3x closer than chance | **exactly zero** |
| shape prior | −2.6% at 100 subjects | **−0.5% n.s.** at 400 |

## 7. Build controls into the test

`crest_paired_test.py` reports anchor error as a control — crest attraction
excludes anchors by construction, so that column MUST be identical between
arms. It was (0.000000mm), which is what makes the null trustworthy rather
than a suspected bug.
