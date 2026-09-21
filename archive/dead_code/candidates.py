"""Candidate-generation + ranking model (Option B). SKELETON -- NOT IMPLEMENTED.

Motivation. A teammate's S57 measured a truth-selected anchor-candidate oracle
of 0.881mm: if candidate positions are proposed and the correct one is always
chosen, error lands near 0.88mm. Our current Stage 1b regresses a correction
vector from a single patch -- it has no candidate set and nothing to rank, which
is a structural reason it plateaus near 1.59mm. This module targets that oracle
instead.

Shape of the pipeline (this part is settled):

    registration  -->  generate_candidates  -->  candidate_features
                                                        |
                                                        v
                                            score_candidates (learned)
                                                        |
                                                        v
                                                select_position

Registration is retained as the region proposer. Everything downstream of it
is new. Infrastructure (dataset, canonical frame, splits, metrics, CV harness)
is shared with the existing pipeline so results stay comparable to 1.5931mm.

EVERY function below is deliberately unimplemented. Each docstring lists the
decisions that must be made before it can be written. Nothing here chooses a
number, an algorithm, a feature, or an architecture.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- generation

def generate_candidates(mesh, seed_position: np.ndarray, anchor_idx: int):
    """Propose candidate positions for one anchor, near `seed_position`.

    OPEN DECISIONS:
      - Source of candidates: curvature extrema? ridge/corridor vertices?
        uniform surface sampling? multi-scale extrema? something else?
      - Candidates on mesh vertices only, or anywhere on the triangle surface?
      - How many per anchor, and fixed count or variable?
      - Search radius around the seed.
      - Whether the candidate rule differs per anchor / per contour.

    ACCEPTANCE CRITERION (decided): the candidate set must be validated with
    scripts/surface_recall.py -- if truth is not close to SOME candidate, no
    ranker can recover it. Target tolerance is itself an open decision.

    Returns: (N, 3) candidate positions.
    """
    raise NotImplementedError("awaiting decisions -- see docstring")


# ------------------------------------------------------------------ features

def candidate_features(mesh, candidates: np.ndarray, seed_position: np.ndarray,
                        anchor_idx: int):
    """Describe each candidate for the ranker.

    OPEN DECISIONS:
      - Per-candidate local descriptor: reuse the existing 7-feature patch
        (256 pts, 7mm) unchanged? a different radius / point count? a different
        feature set entirely?
      - Whether candidates are described independently, or with context about
        the other candidates (relative rank, offsets between them).
      - Whether displacement from the registration seed is an input feature.
      - Whether neighbouring anchors' candidates are visible (joint context).

    Returns: feature tensor for the ranker; exact shape depends on the above.
    """
    raise NotImplementedError("awaiting decisions -- see docstring")


# ------------------------------------------------------------------- ranking

class CandidateRanker(nn.Module):
    """Scores candidates for one anchor.

    OPEN DECISIONS:
      - Architecture family. Note two measured constraints: our attention
        pooling underperformed max pooling, and in the teammate's comparison
        deep models lost to classical statistical ones (best deep 2.55mm vs
        best classical 2.00mm). Extra capacity also stopped paying at their
        error level (~1.55mm).
      - Whether scoring is per-candidate independent, or set-wise (candidates
        compared against each other in one forward pass).
      - Whether one shared model handles all anchors (as now, via an embedding)
        or the S54 pattern applies: shared trunk, separate per-contour heads.
        Note our 4 fully-independent per-contour nets were +11.5% WORSE; S54's
        shared-trunk variant is the version that worked.
      - Capacity / layer sizes.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError("awaiting decisions -- see docstring")

    def forward(self, features):
        raise NotImplementedError("awaiting decisions -- see docstring")


def score_candidates(model: CandidateRanker, features):
    """Run the ranker. Returns one score per candidate."""
    raise NotImplementedError("awaiting decisions -- see docstring")


# ----------------------------------------------------------------- selection

def select_position(candidates: np.ndarray, scores):
    """Collapse scored candidates into one predicted position.

    OPEN DECISIONS:
      - argmax (hard pick), softmax-weighted average (soft), or top-k average?
      - Hard picking caps accuracy at the candidate resolution; soft averaging
        can land between candidates but may drift off-surface.
      - Whether to re-project the result onto the mesh surface afterwards.
    """
    raise NotImplementedError("awaiting decisions -- see docstring")


# ------------------------------------------------------------------ training

def generate_ranking_examples(dataset, template, subject_ids, **kwargs):
    """Build training data for the ranker, leakage-free.

    OPEN DECISIONS:
      - Target formulation: classify the nearest-to-truth candidate?
        regress each candidate's distance to truth? a ranking/contrastive loss
        over the candidate set?
      - Loss function.
      - How to handle ears where NO candidate is near truth.
      - Whether augmentation (jitter/dropout) carries over from the current
        pipeline, and at what magnitude.
    """
    raise NotImplementedError("awaiting decisions -- see docstring")


def train_ranker(examples, **kwargs):
    """Train the ranker.

    OPEN DECISIONS: epochs, batch size, learning rate, weight decay,
    optimiser, ensembling (and how many seeds).

    NOTE: torch_seed must be set. Unseeded runs gave a measured ~0.114mm
    noise floor at 24-subject scale, larger than most effects worth detecting.
    """
    raise NotImplementedError("awaiting decisions -- see docstring")
