"""Heatmap (dense per-point) landmark prediction, with a residual offset head.

We currently do DIRECT COORDINATE REGRESSION: a patch goes in, one 3D offset
comes out, supervised by a single target vector. The landmark-localization
literature overwhelmingly predicts a spatial score map and takes its peak --
every uncertainty measure in arXiv:2203.02351 (S-MHA, E-MHA) is heatmap-derived,
which is why two of their three methods were inapplicable to us.

The 3D analogue for our patches: score each of the 256 patch POINTS with the
probability it is the landmark, soft-argmax over point positions, then add a
small regressed residual.

    patch (256 x 7) --> point MLP --> per-point feature
                             |
                      max-pool -> global feature -> broadcast back
                             |
                        per-point score --softmax--> p_i
                             |
        sum_i p_i * position_i   +   residual offset  --> predicted position

Why it may beat regression here:
  - DENSE SUPERVISION: 256 targets per example instead of 3. This pipeline is
    underfit at every scale tested, so more signal per example is the right
    direction.
  - SURFACE-AWARE: the soft-argmax term is a convex combination of surface
    points, so the coarse estimate cannot float off the mesh. GT landmarks lie
    on the surface to 0.0132mm.
  - EXPRESSES AMBIGUITY: a flat distribution means "somewhere along this ridge"
    -- the exact failure mode hit repeatedly. Regression must commit.

THE HULL PROBLEM, and why the residual head exists
--------------------------------------------------
A soft-argmax is confined to the convex hull of the patch points. Measured seed
error is 3.910mm mean / 7.714mm p95 against a 7.0mm patch radius, so for ~10%
of anchors the true landmark lies OUTSIDE the patch and is unreachable by
pooling alone (candidate_diagnostic.py: 16.94% outside at 6mm, 4.17% at 8mm).

Enlarging the patch was rejected on evidence: 12mm measured +44% WORSE for
regression, because sampling density collapses. So instead the model keeps the
7mm patch and adds a residual offset head, giving dense supervision AND
unbounded range.

Settled parameters (chosen deliberately, see git history for the alternatives):
  HEATMAP_SIGMA = 1.5mm   -- ~30-60 points carry meaningful weight, preserving
                             dense supervision while staying localised
                             (mesh edge is 0.808mm)
  loss           = KL on the distribution + euclidean on the decoded position
  decode         = soft-argmax over all points + residual


ROLE IN THE PIPELINE
--------------------
Reading order : 16 of 20   (correction)
Duty          : Alternative Stage-1b head: dense scores, not regression.

Scores all 256 patch points, softmax, soft-argmax over their positions, plus
a residual head so the answer can leave the convex hull of the patch. Gives
256 supervision targets per example instead of 3, and can express ambiguity
along a ridge.

Known issues / status:
  The only architectural change so far measured as a win: -4.9 percent.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_model import CAPACITY_PRESETS, POINT_FEATURE_DIM, TRAIN_DEVICE
from ..foundations.contours import N_ANCHORS

HEATMAP_SIGMA = 1.5          # mm
DECODE_TEMPERATURE = 1.0
LOSS_POSITION_WEIGHT = 1.0   # weight on the euclidean term relative to KL


def heatmap_target(patch_points_xyz: np.ndarray, true_offset: np.ndarray,
                    sigma: float = HEATMAP_SIGMA) -> np.ndarray:
    """Per-point supervision target: Gaussian over distance to the true landmark.

    `patch_points_xyz` are positions RELATIVE to the patch centre, in mm (i.e.
    the raw rel_xyz feature multiplied back up by PATCH_RADIUS). `true_offset`
    is the vector from the patch centre to the true landmark, same units --
    exactly the `target` field the existing example generators already store.

    Returns a (N,) distribution summing to 1. If the true landmark lies outside
    the patch the target still forms correctly; mass simply piles up on the
    boundary points, and the residual head is what covers the remainder.
    """
    d2 = ((patch_points_xyz - true_offset[None, :]) ** 2).sum(axis=1)
    w = np.exp(-d2 / (2.0 * sigma ** 2))
    s = w.sum()
    if s <= 1e-12:                      # degenerate: truth far outside the patch
        w = np.full(len(w), 1.0 / len(w))
    else:
        w = w / s
    return w.astype(np.float32)


class HeatmapCorrector(nn.Module):
    """Scores every patch point, then adds a residual offset.

    Trunk deliberately mirrors PatchCorrector: it is the only architecture
    validated on this data, and every alternative tried (attention pooling,
    multiscale patches, Gaussian curvature, 12mm radius) measured neutral or
    worse. The only structural addition is broadcasting the pooled global
    feature back to each point before scoring -- without it every point is
    scored in isolation and nothing represents "where along the ridge am I",
    which is the limitation this is meant to address.
    """

    def __init__(self, n_anchor_classes: int = N_ANCHORS,
                 point_feature_dim: int = POINT_FEATURE_DIM,
                 capacity: str = "large"):
        super().__init__()
        mlp_h1, mlp_h2, global_dim, embed_dim, head_h1, head_h2 = CAPACITY_PRESETS[capacity]
        self.point_mlp = nn.Sequential(
            nn.Linear(point_feature_dim, mlp_h1), nn.ReLU(),
            nn.Linear(mlp_h1, mlp_h2), nn.ReLU(),
            nn.Linear(mlp_h2, global_dim), nn.ReLU(),
        )
        self.anchor_embed = nn.Embedding(n_anchor_classes, embed_dim)
        # per-point scoring sees: its own feature + pooled global + anchor embedding
        self.score_head = nn.Sequential(
            nn.Linear(global_dim * 2 + embed_dim, head_h1), nn.ReLU(),
            nn.Linear(head_h1, head_h2), nn.ReLU(),
            nn.Linear(head_h2, 1),
        )
        # residual head sees the pooled global + anchor embedding only
        self.residual_head = nn.Sequential(
            nn.Linear(global_dim + embed_dim, head_h1), nn.ReLU(),
            nn.Linear(head_h1, head_h2), nn.ReLU(),
            nn.Linear(head_h2, 3),
        )

    def forward(self, points: torch.Tensor, anchor_class: torch.Tensor):
        """points: (B, N, 7). Returns (scores (B, N), residual (B, 3))."""
        per_point = self.point_mlp(points)                    # (B, N, G)
        global_feat, _ = per_point.max(dim=1)                  # (B, G)
        embed = self.anchor_embed(anchor_class)                # (B, E)

        n = per_point.shape[1]
        ctx = torch.cat([global_feat, embed], dim=1)           # (B, G+E)
        ctx_b = ctx.unsqueeze(1).expand(-1, n, -1)             # (B, N, G+E)
        scores = self.score_head(torch.cat([per_point, ctx_b], dim=2)).squeeze(-1)

        residual = self.residual_head(ctx)                     # (B, 3)
        return scores, residual


def decode_position(points_xyz: torch.Tensor, scores: torch.Tensor,
                     residual: torch.Tensor, temperature: float = DECODE_TEMPERATURE):
    """Soft-argmax over patch points, plus the residual offset.

    points_xyz: (B, N, 3) positions relative to the patch centre, in mm.
    Returns (B, 3) predicted offset from the patch centre.
    """
    p = F.softmax(scores / temperature, dim=1).unsqueeze(-1)   # (B, N, 1)
    soft = (points_xyz * p).sum(dim=1)                          # (B, 3)
    return soft + residual


def heatmap_loss(scores: torch.Tensor, target: torch.Tensor,
                  decoded: torch.Tensor, true_offset: torch.Tensor,
                  position_weight: float = LOSS_POSITION_WEIGHT,
                  reduction: str = "mean"):
    """KL(target || predicted distribution) + euclidean on the decoded position.

    The KL term trains the whole distribution (this is where the dense
    supervision enters); the euclidean term trains what is actually scored at
    inference. Using either alone loses one of the two.
    """
    logp = F.log_softmax(scores, dim=1)
    kl = F.kl_div(logp, target, reduction="none").sum(dim=1)    # (B,)
    pos = torch.linalg.norm(decoded - true_offset, dim=1)       # (B,)
    per_example = kl + position_weight * pos
    if reduction == "none":
        return per_example
    return per_example.mean()


# ---------------------------------------------------------------- training

class _HeatmapDataset(torch.utils.data.Dataset):
    """Reuses the existing example dicts unchanged.

    The heatmap target is derived on the fly from fields already stored:
    `points[:, :3] * PATCH_RADIUS` gives positions relative to the patch centre
    in mm, and `target` is the offset from that centre to the true landmark.
    So no change to example generation is required.
    """

    def __init__(self, examples, sigma=HEATMAP_SIGMA):
        from .patch_features import PATCH_RADIUS
        self.ex = examples
        self.sigma = sigma
        self.scale = PATCH_RADIUS

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        pts = np.asarray(e["points"], dtype=np.float32)
        xyz = pts[:, :3] * self.scale
        tgt = np.asarray(e["target"], dtype=np.float32)
        hm = heatmap_target(xyz, tgt, self.sigma)
        return (torch.from_numpy(pts), torch.from_numpy(xyz),
                torch.tensor(int(e["anchor_class"])), torch.from_numpy(hm),
                torch.from_numpy(tgt), torch.tensor(float(e.get("loss_weight", 1.0))))


def train_heatmap_model(examples, n_epochs: int = 150, batch_size: int = 64,
                         lr: float = 1e-3, n_classes: int = N_ANCHORS,
                         capacity: str = "large", weight_decay: float = 1e-4,
                         sigma: float = HEATMAP_SIGMA,
                         position_weight: float = LOSS_POSITION_WEIGHT,
                         device=None, verbose: bool = True,
                         torch_seed: int | None = None, optimizer: str = "adam") -> HeatmapCorrector:
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    device = device or TRAIN_DEVICE
    feat_dim = int(np.asarray(examples[0]["points"]).shape[-1])
    model = HeatmapCorrector(n_anchor_classes=n_classes, point_feature_dim=feat_dim,
                              capacity=capacity).to(device)
    from .anchor_model import make_optimizer
    opt = make_optimizer(model.parameters(), optimizer, lr, weight_decay)
    pin = device.type == "cuda"
    loader = torch.utils.data.DataLoader(_HeatmapDataset(examples, sigma),
                                          batch_size=batch_size, shuffle=True, pin_memory=pin)

    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for pts, xyz, cls, hm, tgt, w in loader:
            pts, xyz = pts.to(device, non_blocking=pin), xyz.to(device, non_blocking=pin)
            cls, hm = cls.to(device, non_blocking=pin), hm.to(device, non_blocking=pin)
            tgt, w = tgt.to(device, non_blocking=pin), w.to(device, non_blocking=pin)
            opt.zero_grad()
            scores, resid = model(pts, cls)
            dec = decode_position(xyz, scores, resid)
            per = heatmap_loss(scores, hm, dec, tgt, position_weight, reduction="none")
            loss = (per * w).mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(tgt)
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d}  train_heatmap={total / len(examples):.4f}")

    return model.to("cpu").eval()


def train_heatmap_ensemble(examples, n_models: int = 3, base_seed: int | None = None, **kw) -> list:
    """N independently-initialised heatmap models.

    NOTE: do not pass a fixed torch_seed here -- it would be forwarded to every
    member, giving N identical models and collapsing the ensemble.
    """
    # base_seed gives each member its own seed: reproducible AND diverse
    return [train_heatmap_model(examples, torch_seed=(None if base_seed is None else base_seed + 1000 * i), **kw)
            for i in range(n_models)]


# --------------------------------------------------------------- inference

@torch.no_grad()
def predict_heatmap_correction(model, patch: np.ndarray, anchor_class: int) -> np.ndarray:
    """Drop-in replacement for anchor_model.predict_correction."""
    from .patch_features import PATCH_RADIUS
    if isinstance(model, list):
        return np.mean([predict_heatmap_correction(m, patch, anchor_class) for m in model],
                        axis=0)
    pts = torch.from_numpy(np.asarray(patch, dtype=np.float32)).unsqueeze(0)
    xyz = pts[:, :, :3] * PATCH_RADIUS
    cls = torch.tensor([int(anchor_class)], dtype=torch.long)
    scores, resid = model(pts, cls)
    return decode_position(xyz, scores, resid)[0].numpy()


def is_heatmap(model) -> bool:
    """True if `model` (or an ensemble of them) is a HeatmapCorrector."""
    if isinstance(model, list):
        return bool(model) and isinstance(model[0], HeatmapCorrector)
    return isinstance(model, HeatmapCorrector)


def predict_any_correction(model, patch: np.ndarray, anchor_class: int, ctx=None) -> np.ndarray:
    """Single dispatch point for BOTH model families.

    Heatmap models return (scores, residual) rather than a correction tensor,
    so calling predict_correction on one raises. Every site that turns a patch
    into a correction must route through here -- a previous version dispatched
    correctly in pipeline._predict_at but not in
    training_data.generate_polish_examples, which crashed mid-run.
    """
    if is_heatmap(model):
        return predict_heatmap_correction(model, patch, anchor_class)
    from .anchor_model import predict_correction, predict_correction_ensemble
    if isinstance(model, list):
        return predict_correction_ensemble(model, patch, anchor_class, ctx)
    return predict_correction(model, patch, anchor_class, ctx)
