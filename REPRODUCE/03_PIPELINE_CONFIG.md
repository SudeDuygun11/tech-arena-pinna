# Every pipeline constant

Changing any of these changes the results. All are module-level constants
unless a CLI flag is noted.

## Landmark structure — `src/contours.py`

```
ANCHOR_INDICES = [0, 6, 22, 24, 25, 33, 42, 46, 50, 54, 55, 64, 74, 75, 84]   (15)

outer_helix         range (0, 25)    anchors [0, 6, 22, 24]          25 pts, 4 anchors
concha_outline      range (25, 55)   anchors [25, 33, 42, 46, 50, 54] 30 pts, 6 anchors
inner_helix         range (55, 75)   anchors [55, 64, 74]            20 pts, 3 anchors
superior_antihelix  range (75, 85)   anchors [75, 84]                10 pts, 2 anchors
```

Index 74 is a "derived continuation" endpoint, structurally unlike the other 14
true anatomical anchors.

## Registration — `src/registration.py`

| constant | value |
|---|---|
| `N_CONTROL_SURFACE_POINTS` | 450 |
| `TPS_SMOOTHING` | 4.0 |
| `RIGID_ICP_ITERS` | 8 |
| `NONRIGID_ICP_ITERS` | 6 |
| `TRIM_FRACTION` | 0.15 |
| CPD params (unused by default) | beta 3.0, lambda 3.0, outlier 0.1, iters 50 |

`--k-references` (CLI, default 5; **all reported results use 7**).

## Patch features — `src/patch_features.py`

| constant | value |
|---|---|
| **`PATCH_RADIUS`** | **7.0 mm** |
| **`PATCH_N_POINTS`** | **256** |
| `POINT_FEATURE_DIM` | 7 = rel_xyz(3) + normal(3) + curvature(1) |

`rel_xyz` is divided by `radius`; curvature is a uniform-Laplacian proxy dotted
with the vertex normal. CLI overrides: `--patch-radius`, `--patch-points`,
`--geodesic-patch`, `--gaussian-curvature` (8th feature).

12mm was tested and is **+44% worse**. 7mm is not arbitrary.

## Augmentation — `src/training_data.py`

| constant | value |
|---|---|
| `AUGMENT_JITTER_STD` | 1.5 mm (on the patch CENTRE, simulating registration error) |
| `AUGMENT_DROPOUT_P` | 0.15 (probability of a dropout mask; masked points zeroed at rate 0.3) |

Polish-stage jitter is `AUGMENT_JITTER_STD * 0.5`. Increasing jitter was tested
and is worse.

## Network — `src/anchor_model.py`

```
CAPACITY_PRESETS = (mlp_h1, mlp_h2, global_dim, embed_dim, head_h1, head_h2)
  small   ( 32,  64,  128, 16,  64,  32)
  large   ( 64, 128,  256, 32, 128,  64)   <- all headline results
  xlarge  (128, 256,  512, 64, 256, 128)
  xxlarge (256, 512, 1024,128, 512, 256)
```

Architecture: point-wise MLP -> max-pool -> concat anchor embedding -> MLP head
-> 3D correction. Anchor stage uses 15 classes; polish stage uses 85.

Training defaults: Adam, lr 1e-3, batch 64, weight_decay 1e-4, SmoothL1 loss.

## Interpolation — `src/interpolation.py`

| constant | value |
|---|---|
| `DEFAULT_SNAP_RADIUS` | 3.0 mm |

Crest-attraction constants exist but are a measured null; leave `--crest-attract` off.

## Shape prior — `src/joint_fit.py`

| constant | value |
|---|---|
| `DEFAULT_N_MODES` | 45 |
| `DEFAULT_STRENGTH` | 0.05 |
| Procrustes | rigid only (rotation+translation, **NOT scale** — ear size is real signal) |

Measured null at 400 ears; documented for completeness.
