"""Surface-query transformer: global tokens + per-landmark queries. SKELETON.

WHY THIS ARCHITECTURE
---------------------
A teammate's S111 "surface-query transformer" scores 1.2822mm on 160
subjects/320 ears, versus our 1.5931mm on 200/400 -- roughly 20% better, and
UNIFORMLY better across all four contours (outer 1.401 vs 1.828, concha 0.979
vs 1.146, inner 1.585 vs 1.999, superior 1.264 vs 1.552). A uniform gap that
size is a method difference, not tuning.

Their earlier S42 in the same family was "384 physical surface tokens with 85
anatomy-conditioned queries", and went 2.5522 -> 1.2822mm through development.
Their portfolio of four different methods beats S111 alone by only 0.003mm, so
the architecture is where essentially all of their advantage lives.

THE STRUCTURAL DIFFERENCE
-------------------------
                       ours                    surface-query
  input        15 separate 7mm patches   ~384 tokens covering the WHOLE ear
  prediction   15 anchors -> interpolate  85 learnable queries, one per
               the other 70               landmark, cross-attending the surface
  context      ~1/10 of the ear per pt    the entire ear, for every point
  interpolation  required                 none -- all 85 predicted directly

This dissolves the single failure mode that defeated four separate attempts in
this project. Local geometry can say "ridge-like" but never "WHERE along the
ridge" -- which is why crest attraction produced exactly zero, candidate
ranking landed at 5.5mm against a 0.53mm oracle, the shape prior measured null,
and a 12mm patch was +44% worse. A query attending over global tokens has the
positional context that a 7mm window structurally cannot.

    ear surface  --> tokenizer --> S tokens (S x D)
                                        |
                                   self-attention (tokens see each other)
                                        |
    85 learned queries (85 x D) --> cross-attention over tokens
                                        |
                                   per-query head --> 85 x 3 positions

CAUTION, AND WHY THIS IS NOT REFUTED BY OUR ATTENTION RESULT
-----------------------------------------------------------
We tested "attention pooling" and it LOST (+0.16mm, the only one of four
variants clearing the noise floor). But that was attention over the 256 points
INSIDE one patch, replacing a max-pool. It is a different mechanism from
queries attending over global surface tokens, and is not evidence against this.

The real risk is data volume: 200 subjects / 400 ears is small for a
transformer, and this project has repeatedly found added capacity neutral or
harmful. Against that: the teammate trained this family successfully on 160
subjects, which is FEWER than we have.

Every function below is deliberately unimplemented; each docstring lists the
decisions required.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def tokenize_ear(mesh, template_center: np.ndarray, template_radius: float):
    """Turn one ear into a fixed-size set of surface tokens.

    OPEN DECISIONS:
      - Token count. S42 used 384. Ours must cover a ~75mm ear span.
      - Sampling: farthest-point sampling (even coverage -- FPS measured best
        for coverage in candidate_selection_diagnostic.py, 0.542mm vs random
        0.584mm), uniform random, or curvature-weighted (measured WORSE for
        coverage: curvature_top 1.094mm).
      - Token features. Our 7 per-point features (rel_xyz, normal, curvature)
        are validated; four attempts to extend them landed within noise. But
        tokens need POSITION IN THE EAR, not position relative to a patch
        centre -- the whole point is global context, so the coordinate frame
        must be ear-level, not patch-level.
      - Frame: raw canonical coordinates, or normalised by ear extent? Note
        ear size (66-82mm span) is real anatomical signal and the metric is in
        millimetres, so scale must NOT be normalised away (this is why
        unit-sphere normalisation was rejected).
      - Whether each token also carries a small local patch descriptor, giving
        the model both scales at once.

    Returns: (S, F) token features and (S, 3) token positions.
    """
    raise NotImplementedError("awaiting decisions -- see docstring")


class SurfaceQueryTransformer(nn.Module):
    """Learned per-landmark queries cross-attending over surface tokens.

    OPEN DECISIONS:
      - Depth/width/heads. Small is safer: 200 subjects, and every capacity
        increase past `large` has been neutral or negative here.
      - Whether tokens get self-attention before cross-attention (lets the
        surface representation become context-aware first) or queries attend
        raw tokens directly (cheaper, fewer parameters).
      - Positional encoding for tokens. Their coordinates ARE the position, so
        a learned projection of xyz may suffice, or a Fourier/sinusoidal
        encoding may be needed for high-frequency detail.
      - Output parameterisation, and this is the consequential one:
          (a) absolute position per query;
          (b) offset from the registration prior (keeps the validated
              3.91mm starting point and only learns a residual);
          (c) attention-weighted combination of token positions, i.e. the
              heatmap/soft-argmax idea lifted to global scope -- surface
              constrained by construction, but bounded by token coverage.
        (b) is the most conservative and reuses machinery that works.
      - 85 independent queries, or 15 anchor queries + 70 derived? The 70
        non-anchor points are provably evenly spaced by chord length
        (CV 0.5-1.8%), so predicting them independently discards a
        near-deterministic constraint.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError("awaiting decisions -- see docstring")

    def forward(self, token_features: torch.Tensor, token_positions: torch.Tensor):
        raise NotImplementedError("awaiting decisions -- see docstring")


def build_training_examples(dataset, template, subject_ids, **kwargs):
    """One example per EAR, not per landmark.

    This is a structural break from the current data pipeline, which produces
    15 (patch, anchor_class, target) rows per ear. Here each ear is a single
    example with all 85 targets, so the effective dataset size drops from
    ~6,000 rows to ~400. That is a real risk for a transformer and makes
    augmentation more important, not less.

    OPEN DECISIONS:
      - Augmentation. Existing jitter (1.5mm on patch centres) does not apply;
        there is no patch centre. Candidates: small rotations, non-rigid
        deformation, token resampling (a fresh FPS draw per epoch is free
        augmentation and costs nothing).
      - Whether the registration prior is an input (for output mode (b)).
      - Leakage control is unchanged: templates must come from other folds.
    """
    raise NotImplementedError("awaiting decisions -- see docstring")


def train_surface_query(examples, **kwargs):
    """Training loop.

    OPEN DECISIONS: optimiser, LR schedule (transformers usually want warmup,
    unlike the flat Adam used elsewhere here), epochs, batch size in EARS,
    loss (SmoothL1 per landmark, as validated -- note Wing loss was +7.1%
    worse here), and whether per-contour loss weights apply.

    Use derive_seed() from anchor_model for per-member seeds if ensembling.
    """
    raise NotImplementedError("awaiting decisions -- see docstring")
