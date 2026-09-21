"""Stage-1 learned residual-correction network.

Given the registration prior's predicted position for one anchor and a local
surface patch around it, predict a 3D correction vector so that
prediction + correction ~= true anchor position. A single shared network
handles all 15 anchor classes via a learned embedding, and both ears via the
canonical left-ear mirroring done upstream (see geometry.mirror_y).


ROLE IN THE PIPELINE
--------------------
Reading order : 9 of 20   (correction)
Duty          : The PointNet-style correction network and its training loop.

Shared per-point MLP, max-pool over points, concatenate a learned per-anchor
embedding, MLP head, 3D correction. Max-pooling is what makes it
permutation-invariant over an unordered patch. derive_seed gives each
ensemble member a distinct reproducible seed.

Known issues / status:
  Adam lr 1e-3, batch 64, weight_decay 1e-4 are untouched defaults and were
  never swept.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset

from ..foundations.contours import N_ANCHORS

POINT_FEATURE_DIM = 7
EMBED_DIM = 16
GLOBAL_FEATURE_DIM = 128

# Training is batched (many examples at once) so a GPU helps there. Inference
# (predict_correction) is called one tiny patch at a time inside per-point
# loops, where GPU kernel-launch/transfer overhead would dominate the actual
# compute -- so trained models are moved back to CPU before being used for
# inference (see train_model / predict_correction below).
TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CAPACITY_PRESETS = {
    # (mlp_h1, mlp_h2, global_dim, embed_dim, head_h1, head_h2)
    "small": (32, 64, 128, 16, 64, 32),      # current/default architecture
    "large": (64, 128, 256, 32, 128, 64),    # ~4x params; -12.8% in an ISOLATED test
    # NOTE: the -12.8% / -19.3% figures come from small isolated runs, NOT from
    # full-scale cross-validation. SOLUTION_README lists `xlarge` under "never
    # validated at full CV scale". Do not quote these as validated results.
    "xlarge": (128, 256, 512, 64, 256, 128),  # ~16x params; -19.3% ISOLATED, unvalidated at scale
    "xxlarge": (256, 512, 1024, 128, 512, 256),  # ~64x parameters -- probing for the ceiling
}


class PatchCorrector(nn.Module):
    def __init__(self, n_anchor_classes: int = N_ANCHORS,
                 point_feature_dim: int = POINT_FEATURE_DIM,
                 embed_dim: int = EMBED_DIM,
                 global_dim: int = GLOBAL_FEATURE_DIM,
                 mlp_h1: int = 32, mlp_h2: int = 64,
                 head_h1: int = 64, head_h2: int = 32,
                 pooling: str = "max", ctx_dim: int = 0):
        super().__init__()
        self.pooling = pooling
        self.ctx_dim = ctx_dim
        pool_mult = 2 if pooling == "max_mean" else 1
        self.point_mlp = nn.Sequential(
            nn.Linear(point_feature_dim, mlp_h1), nn.ReLU(),
            nn.Linear(mlp_h1, mlp_h2), nn.ReLU(),
            nn.Linear(mlp_h2, global_dim), nn.ReLU(),
        )
        if pooling == "attention":
            # Lightweight single-layer attention pooling: one learned scoring
            # linear layer produces a softmax weight per point, replacing the
            # fixed max/mean rule with a learned weighted average. This is
            # deliberately NOT a full self-attention block (no Q/K/V
            # projections, no multi-head, no point-to-point interaction) --
            # just `global_dim` extra parameters, chosen to stay cheap given
            # our data size after the Transformer-vs-data-size discussion.
            self.attn_score = nn.Linear(global_dim, 1)
        self.anchor_embed = nn.Embedding(n_anchor_classes, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(global_dim * pool_mult + embed_dim + ctx_dim, head_h1), nn.ReLU(),
            nn.Linear(head_h1, head_h2), nn.ReLU(),
            nn.Linear(head_h2, 3),
        )

    @classmethod
    def from_capacity(cls, capacity: str, n_anchor_classes: int = N_ANCHORS,
                       pooling: str = "max",
                       point_feature_dim: int = POINT_FEATURE_DIM, ctx_dim: int = 0) -> "PatchCorrector":
        mlp_h1, mlp_h2, global_dim, embed_dim, head_h1, head_h2 = CAPACITY_PRESETS[capacity]
        return cls(n_anchor_classes=n_anchor_classes, embed_dim=embed_dim, global_dim=global_dim,
                    mlp_h1=mlp_h1, mlp_h2=mlp_h2, head_h1=head_h1, head_h2=head_h2, pooling=pooling,
                    point_feature_dim=point_feature_dim, ctx_dim=ctx_dim)

    def forward(self, points: torch.Tensor, anchor_class: torch.Tensor,
                ctx: torch.Tensor | None = None) -> torch.Tensor:
        """points: (B, N, 7) patch features. anchor_class: (B,) long indices in [0, 15).
        Returns (B, 3) predicted correction vectors (same units as PATCH_RADIUS-scaled input,
        i.e. already in raw mesh units since rel_xyz normalization is undone by training targets
        being expressed in raw units -- see train_anchor_model.py)."""
        per_point = self.point_mlp(points)  # (B, N, global_dim)
        if self.pooling == "max_mean":
            max_feat, _ = per_point.max(dim=1)
            mean_feat = per_point.mean(dim=1)
            global_feat = torch.cat([max_feat, mean_feat], dim=1)  # (B, 2*global_dim)
        elif self.pooling == "attention":
            scores = self.attn_score(per_point)  # (B, N, 1)
            weights = torch.softmax(scores, dim=1)  # (B, N, 1)
            global_feat = (per_point * weights).sum(dim=1)  # (B, global_dim)
        else:
            global_feat, _ = per_point.max(dim=1)  # (B, global_dim)
        embed = self.anchor_embed(anchor_class)  # (B, embed_dim)
        parts = [global_feat, embed]
        if self.ctx_dim > 0:
            # reference-agreement context (see registration/refbank.py); zeros if absent
            parts.append(ctx if ctx is not None else global_feat.new_zeros(global_feat.shape[0], self.ctx_dim))
        return self.head(torch.cat(parts, dim=1))


class WingLoss(nn.Module):
    """Wing loss (Feng et al. 2018), for direct coordinate/vector regression
    (not heatmaps -- see Adaptive Wing for the heatmap variant).

    SmoothL1's gradient shrinks as error shrinks, giving less training
    pressure exactly where we now need it most (average error is already a
    few mm). Wing loss instead amplifies gradient in the small-to-medium
    error range (below `omega`) via a log term, while staying linear (robust
    to outliers) above it -- omega/epsilon are set in mm to match our error
    scale, not the original paper's pixel-scale defaults.
    """

    def __init__(self, omega: float = 3.0, epsilon: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.reduction = reduction
        self.C = omega - omega * math.log(1 + omega / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        small = diff < self.omega
        loss = torch.where(small, self.omega * torch.log(1 + diff / self.epsilon), diff - self.C)
        return loss if self.reduction == "none" else loss.mean()


def make_optimizer(params, name: str = "adam", lr: float = 1e-3, weight_decay: float = 1e-4):
    """Adam couples weight decay into the adaptive gradient (L2 regularisation,
    scaled per parameter); AdamW applies it decoupled from the gradient
    (Loshchilov & Hutter, ICLR 2019), which is what 'weight decay' intends.
    AdamW's weight_decay is conventionally ~1e-2, not Adam's 1e-4."""
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def make_loss(loss_type: str = "smooth_l1", reduction: str = "mean",
              wing_omega: float = 3.0, wing_epsilon: float = 1.0) -> nn.Module:
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss(reduction=reduction)
    if loss_type == "wing":
        return WingLoss(omega=wing_omega, epsilon=wing_epsilon, reduction=reduction)
    raise ValueError(f"unknown loss_type {loss_type!r}")


def _random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-12
    ang = np.radians(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def augment_patch(points: np.ndarray, target: np.ndarray, rng: np.random.Generator,
                  rotate_deg: float, scale_frac: float, dropout_frac: float = 0.0):
    """Consistent small rotation + isotropic scale of one training example.
    Channel layout from patch_features: 0-2 xyz, 3-5 normal; then per mode --
    base: curvature (6); curvedness: shape index (6), curvedness (7);
    rich: shape index (6), curvedness (7), ridge tensor (8-13), enclosure (14)."""
    from .patch_features import point_features
    p = points.copy(); t = target.copy()
    real = np.abs(p).sum(1) > 0
    if rotate_deg > 0:
        R = _random_rotation(rng, rotate_deg)
        p[real, 0:3] = p[real, 0:3] @ R.T
        p[real, 3:6] = p[real, 3:6] @ R.T
        t = R @ t
        if point_features() == "rich" and p.shape[1] >= 14:
            c = p[real, 8:14]
            M = np.stack([np.stack([c[:, 0], c[:, 3], c[:, 4]], 1), np.stack([c[:, 3], c[:, 1], c[:, 5]], 1),
                          np.stack([c[:, 4], c[:, 5], c[:, 2]], 1)], 1)
            M = R @ M @ R.T
            p[real, 8:14] = np.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2], M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]], 1)
    if scale_frac > 0:
        s = rng.uniform(1 - scale_frac, 1 + scale_frac)
        p[real, 0:3] *= s; t = t * s
        if point_features() in ("curvedness", "rich"):
            p[real, 7] = np.clip(p[real, 7] / s, 0.0, 3.0)
    if dropout_frac > 0:
        # zeroed rows are exactly what padding looks like, so the network already
        # treats them as absent points
        idx = np.flatnonzero(real)
        n_drop = int(len(idx) * rng.uniform(0.0, dropout_frac))
        if n_drop > 0:
            p[rng.choice(idx, n_drop, replace=False)] = 0.0
    return p.astype(np.float32), t.astype(np.float32)


class _ExampleDataset(TorchDataset):
    def __init__(self, examples: list[dict], rotate_deg: float = 0.0, scale_frac: float = 0.0,
                 dropout_frac: float = 0.0):
        self.examples = examples
        self.rotate_deg, self.scale_frac, self.dropout_frac = rotate_deg, scale_frac, dropout_frac
        # seeded from torch's generator, so --torch-seed makes augmentation replayable
        self.rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        weight = ex.get("loss_weight", 1.0)
        pts, tgt = ex["points"], ex["target"]
        if self.rotate_deg > 0 or self.scale_frac > 0 or self.dropout_frac > 0:
            pts, tgt = augment_patch(pts, tgt, self.rng, self.rotate_deg, self.scale_frac,
                                     self.dropout_frac)
        ctx = ex.get("ctx")
        ctx = torch.from_numpy(np.asarray(ctx, dtype=np.float32)) if ctx is not None else torch.zeros(0)
        return (torch.from_numpy(np.ascontiguousarray(pts)), torch.tensor(ex["anchor_class"], dtype=torch.long),
                torch.from_numpy(np.ascontiguousarray(tgt)), torch.tensor(weight, dtype=torch.float32), ctx)


def train_model(examples: list[dict], n_epochs: int = 40, batch_size: int = 64,
                 lr: float = 1e-3, val_examples: list[dict] | None = None,
                 verbose: bool = True, n_classes: int = N_ANCHORS,
                 device: torch.device | None = None, loss_type: str = "smooth_l1",
                 capacity: str = "small", weight_decay: float = 1e-4,
                 pooling: str = "max", torch_seed: int | None = None,
                 optimizer: str = "adam", loss_weighting: str = "none",
                 augment_rotate: float = 0.0, augment_scale: float = 0.0,
                 augment_dropout: float = 0.0, lr_schedule: str = "constant",
                 swa_start: float = 0.0, wing_omega: float = 3.0,
                 wing_epsilon: float = 1.0) -> PatchCorrector:
    # Without this, torch's global RNG (weight init + batch shuffling) is
    # unseeded, so every run trains a DIFFERENT network and any A/B comparison
    # silently measures (change + training variance) with no estimate of the
    # variance term. Pass torch_seed to make a run reproducible; vary it
    # deliberately across repeats to measure that variance instead of ignoring it.
    # NOTE: train_multi_seed_ensemble relies on consecutive calls differing, so
    # it must NOT pass a fixed torch_seed to every member.
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    device = device or TRAIN_DEVICE
    # infer input width from the data, so adding/removing per-point features
    # (e.g. Gaussian curvature) needs no flag threading through the training stack
    feat_dim = int(np.asarray(examples[0]["points"]).shape[-1])
    ctx_dim = int(np.asarray(examples[0]["ctx"]).shape[-1]) if examples[0].get("ctx") is not None else 0
    model = PatchCorrector.from_capacity(capacity, n_anchor_classes=n_classes, pooling=pooling,
                                          point_feature_dim=feat_dim, ctx_dim=ctx_dim).to(device)
    params = list(model.parameters())
    log_var = None
    if loss_weighting == "uncertainty":
        # Kendall et al. 2018: one learnable log-variance per anchor class
        log_var = torch.nn.Parameter(torch.zeros(n_classes, device=device))
    opt = make_optimizer(params, optimizer, lr, weight_decay)
    if log_var is not None:
        opt.add_param_group({"params": [log_var], "weight_decay": 0.0})
    # reduction='none' so per-example loss_weight (e.g. per-contour) can be applied before averaging
    loss_fn = make_loss(loss_type, reduction="none", wing_omega=wing_omega, wing_epsilon=wing_epsilon)
    pin = device.type == "cuda"
    loader = DataLoader(_ExampleDataset(examples, augment_rotate, augment_scale, augment_dropout),
                         batch_size=batch_size, shuffle=True, pin_memory=pin)
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)
    elif lr_schedule == "constant":
        scheduler = None
    else:
        raise ValueError(f"unknown lr_schedule {lr_schedule!r}")
    swa_model = None
    swa_from = int(round(swa_start * n_epochs)) if swa_start > 0 else None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for points, anchor_class, target, weight, ctx in loader:
            points = points.to(device, non_blocking=pin)
            anchor_class = anchor_class.to(device, non_blocking=pin)
            target = target.to(device, non_blocking=pin)
            weight = weight.to(device, non_blocking=pin)
            opt.zero_grad()
            pred = model(points, anchor_class, ctx.to(device, non_blocking=pin) if ctx_dim else None)
            per_example = loss_fn(pred, target).mean(dim=1)  # (B,)
            if log_var is not None:
                s = log_var[anchor_class]
                per_example = torch.exp(-s) * per_example + s
            loss = (per_example * weight).mean()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(target)
        train_loss = total_loss / len(examples)
        if scheduler is not None:
            scheduler.step()
        if swa_from is not None and epoch >= swa_from:
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
            swa_model.update_parameters(model)  # the constructor copies but does not count

        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            msg = f"  epoch {epoch:3d}  train_{loss_type}={train_loss:.4f}"
            if val_examples:
                msg += f"  val_mean_dist={evaluate_correction_error(model, val_examples):.4f}"
            print(msg)

    if log_var is not None and verbose:
        w = torch.exp(-log_var).detach().cpu().numpy()
        print("  learned per-class loss weights exp(-s): " + " ".join(f"{v:.2f}" for v in w))
    if swa_model is not None:
        # PatchCorrector has no BatchNorm, so the averaged weights need no recalibration
        model.load_state_dict(swa_model.module.state_dict())
        if verbose:
            print(f"  SWA: averaged {int(swa_model.n_averaged)} epoch snapshots")
    return model.to("cpu")  # inference is single-patch-at-a-time; CPU avoids per-call GPU overhead


@torch.no_grad()
def evaluate_correction_error(model: PatchCorrector, examples: list[dict]) -> float:
    """Mean Euclidean *residual* error after applying the model's correction
    (i.e. how far the corrected anchor still is from ground truth)."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(_ExampleDataset(examples), batch_size=256, shuffle=False)
    dists = []
    for points, anchor_class, target, _weight, ctx in loader:
        points, anchor_class, target = points.to(device), anchor_class.to(device), target.to(device)
        pred = model(points, anchor_class, ctx.to(device) if model.ctx_dim else None)
        dists.append(torch.linalg.norm(pred - target, dim=1))
    return torch.cat(dists).mean().item()


@torch.no_grad()
def predict_correction(model: PatchCorrector, points: np.ndarray, anchor_class: int,
                       ctx: np.ndarray | None = None) -> np.ndarray:
    model.eval()
    p = torch.from_numpy(points).unsqueeze(0)
    c = torch.tensor([anchor_class], dtype=torch.long)
    x = torch.from_numpy(np.asarray(ctx, dtype=np.float32)).unsqueeze(0) if ctx is not None else None
    return model(p, c, x).squeeze(0).numpy()


def train_snapshot_ensemble(examples: list[dict], n_cycles: int = 4, epochs_per_cycle: int = 40,
                             batch_size: int = 64, max_lr: float = 1e-3, val_examples=None,
                             verbose: bool = True, n_classes: int = N_ANCHORS,
                             device: torch.device | None = None,
                             loss_type: str = "smooth_l1", capacity: str = "small") -> list:
    """Snapshot ensembling (Huang et al. 2017): one training run, cosine-annealed
    learning rate that resets to `max_lr` at the start of each cycle and decays
    to ~0 by the end of it -- a snapshot is saved at the end of every cycle,
    each one a different local minimum. "M models for the training cost of 1."
    Returns a list of `n_cycles` PatchCorrector models (all on CPU)."""
    device = device or TRAIN_DEVICE
    feat_dim = int(np.asarray(examples[0]["points"]).shape[-1])
    model = PatchCorrector.from_capacity(capacity, n_anchor_classes=n_classes,
                                          point_feature_dim=feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=epochs_per_cycle, T_mult=1)
    loss_fn = make_loss(loss_type, reduction="none")
    pin = device.type == "cuda"
    loader = DataLoader(_ExampleDataset(examples), batch_size=batch_size, shuffle=True,
                         pin_memory=pin)

    snapshots = []
    total_epochs = n_cycles * epochs_per_cycle
    for epoch in range(total_epochs):
        lr_used = opt.param_groups[0]["lr"]  # LR this epoch's batches will actually train with
        model.train()
        total_loss = 0.0
        for points, anchor_class, target, weight, _ctx in loader:
            points = points.to(device, non_blocking=pin)
            anchor_class = anchor_class.to(device, non_blocking=pin)
            target = target.to(device, non_blocking=pin)
            weight = weight.to(device, non_blocking=pin)
            opt.zero_grad()
            pred = model(points, anchor_class)
            per_example = loss_fn(pred, target).mean(dim=1)
            loss = (per_example * weight).mean()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(target)
        scheduler.step(epoch + 1)  # advance once per epoch, after training on it
        train_loss = total_loss / len(examples)

        cycle, epoch_in_cycle = divmod(epoch, epochs_per_cycle)
        is_cycle_end = epoch_in_cycle == epochs_per_cycle - 1
        if verbose and (epoch % 5 == 0 or is_cycle_end):
            msg = f"  epoch {epoch:3d} (cycle {cycle + 1}/{n_cycles}) train_{loss_type}={train_loss:.4f} lr={lr_used:.2e}"
            if val_examples:
                msg += f"  val_mean_dist={evaluate_correction_error(model, val_examples):.4f}"
            print(msg)

        if is_cycle_end:
            import copy
            snap = copy.deepcopy(model).to("cpu")
            snap.eval()
            snapshots.append(snap)
            if verbose:
                print(f"    -> saved snapshot {len(snapshots)}/{n_cycles}")

    return snapshots


@torch.no_grad()
def predict_correction_ensemble(models: list, points: np.ndarray, anchor_class: int,
                                ctx: np.ndarray | None = None) -> np.ndarray:
    """Average the correction predicted by each snapshot/ensemble member."""
    preds = [predict_correction(m, points, anchor_class, ctx) for m in models]
    return np.mean(preds, axis=0)


def derive_seed(base_seed: int, outer_fold: int = 0, stage: int = 0, member: int = 0) -> int:
    """Distinct, reproducible seed per (run, fold, stage, ensemble member).

    An ensemble needs its members to DIFFER, but a run needs to be REPLAYABLE.
    Passing one fixed seed to every member satisfies the second and destroys the
    first -- all N members get identical weights and the ensemble collapses
    silently. Omitting seeding entirely (the previous behaviour) satisfies the
    first and destroys the second.

    Deriving a distinct seed per member gives both. `stage` separates the anchor
    stage (0) from the polish stage (1) so they do not share trajectories.
    """
    return int((base_seed * 1_000_003 + outer_fold * 10_007 + stage * 101 + member) % (2 ** 31 - 1))


def train_multi_seed_ensemble(examples: list[dict], n_models: int = 3,
                               base_seed: int | None = None, outer_fold: int = 0,
                               stage: int = 0, **train_model_kwargs) -> list:
    """Train N fully independent models and return them for
    predict_correction_ensemble/_dispatch_correction to average over.

    Unlike snapshot ensembling (one training run, cyclic LR, snapshots share
    an early training trajectory), each member here is trained completely
    independently from scratch -- more diverse, at N x the training cost.

    Members must DIFFER from one another. With `base_seed` given, each gets its
    own derived seed (see derive_seed), so the run is both reproducible and
    diverse. With base_seed=None the previous unseeded behaviour is kept --
    members still differ, but the run cannot be replayed.

    Never pass `torch_seed` in train_model_kwargs: it would forward the SAME
    seed to every member and collapse the ensemble.
    """
    if "torch_seed" in train_model_kwargs:
        raise ValueError(
            "pass base_seed=... to train_multi_seed_ensemble, not torch_seed -- "
            "a single torch_seed would be applied to every member, making them identical")
    verbose = train_model_kwargs.pop("verbose", True)
    models = []
    for i in range(n_models):
        if verbose:
            print(f"  training ensemble member {i + 1}/{n_models}...")
        member_seed = (None if base_seed is None
                       else derive_seed(base_seed, outer_fold, stage, i))
        models.append(train_model(examples, verbose=(verbose and i == 0),
                                   torch_seed=member_seed, **train_model_kwargs))
    return models
