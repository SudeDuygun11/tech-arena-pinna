"""Curve layer: let anchors on the same contour see each other before committing.

WHY THIS EXISTS
---------------
Every anchor is currently predicted ALONE from its own 7mm patch. Nothing lets
anchor 6 know what anchors 0, 22 and 24 decided. But the 15 anchors are not
independent points -- they are the corners of four ordered curves.

Two measurements motivate this directly.

1. ANCHORS ARE DEFINED RELATIONALLY, NOT LOCALLY. Measured over 400 ears, the
   interior anchors sit where the contour TURNS:

       concha_outline   41.1 deg turn at anchors vs 18.7 elsewhere  (2.20x)
       outer_helix      22.7 vs 12.0                                (1.89x)
       inner_helix      17.0 vs 12.7                                (1.34x)

   plus every contour's two endpoints. "This is where the curve bends" is a
   statement ABOUT THE CURVE. A 7mm patch cannot see the curve -- it has no
   idea which way the contour runs or whether it is bending. The information
   an anchor is defined by is structurally absent from its input.

2. THE SAME BLINDNESS EXPLAINS THE 'WRONG RIDGE' FAILURE. A candidate oracle
   measured 0.534mm, but an actual ranker over those candidates got 5.524mm --
   coverage was fine, SELECTION was wrong. A teammate reached the identical
   diagnosis independently: "candidate pools often contain a close point, yet
   learned selectors frequently prefer a denser or more curved WRONG RIDGE.
   This separation -- candidate coverage versus semantic selection -- is now a
   central diagnosis."

   An ear has several near-parallel ridges that look alike within 7mm. But if
   anchors 0, 22 and 24 have settled on the outer rim, an anchor 6 sitting on
   the inner fold is detectably inconsistent WITH ITS OWN CURVE.

This is also why six separate attempts failed (crest attraction, candidate
ranking, shape prior, geodesic patches, 12mm patches, hard-example mining):
each tried to extract a relational property from local evidence.

DESIGN
------
Deliberately small -- hidden 96, 6 heads, 1 layer, copied from a teammate's
S111 spec, which is a validated operating point on 160 subjects (we have 200).
Capacity has been neutral-or-harmful in this project at every size past
`large`, so this is not the place to be generous.

Attention is applied WITHIN each contour, never across contours: an outer-helix
anchor should be constrained by the outer helix, not by the concha. Contours
have 2-6 anchors, so these are very small attention sets.

Output is a RESIDUAL on top of the incoming anchor positions, not a fresh
prediction. The incoming positions are already good (1.66mm); the layer only
has to nudge inconsistent ones back onto their curve. A residual formulation
also means an untrained/zero-initialised layer is the identity, so it cannot
make things worse before it has learned anything.


ROLE IN THE PIPELINE
--------------------
Reading order : 18 of 20   (experimental)
Duty          : Attention among a contour's anchors. MEASURED NULL.

Lets anchors on the same contour see each other before interpolation, aimed
at the wrong-ridge failure.

Known issues / status:
  +1.4 percent on anchors, fold 1 only; fold 2 died on MemoryError. The design
  is known to be wrong: attention was restricted WITHIN each contour, but the
  annotator's definitions are cross-contour (64 is defined by 6, 55 by the
  height of 25, and 33 and 75 sit 2.148mm apart).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS

HIDDEN = 96
N_HEADS = 6
N_LAYERS = 1

# anchors grouped by contour, in curve order, as positions within ANCHOR_INDICES
_POS = {a: i for i, a in enumerate(ANCHOR_INDICES)}
CONTOUR_ANCHOR_SLOTS = {
    name: [_POS[a] for a in sorted(spec["anchors"])]
    for name, spec in CONTOUR_SPECS.items()
}


class CurveLayer(nn.Module):
    """Self-attention among the anchors of each contour, predicting a residual.

    Input : (B, 15, 3) anchor positions, in the order of ANCHOR_INDICES
    Output: (B, 15, 3) adjusted positions
    """

    def __init__(self, hidden: int = HIDDEN, n_heads: int = N_HEADS,
                 n_layers: int = N_LAYERS, use_confidence: bool = False):
        super().__init__()
        self.use_confidence = use_confidence
        in_dim = 3 + (1 if use_confidence else 0)

        # A learned embedding per anchor SLOT carries curve identity and order:
        # which contour this anchor belongs to and where along it it sits. The
        # turning-point structure is exactly an ordered property, so order must
        # be representable.
        self.slot_embed = nn.Embedding(len(ANCHOR_INDICES), hidden)
        self.input_proj = nn.Linear(in_dim, hidden)

        enc = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)

        self.out = nn.Linear(hidden, 3)
        # zero-init the output so an untrained layer is exactly the identity
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, anchors: torch.Tensor, confidence: torch.Tensor | None = None):
        B = anchors.shape[0]
        # Work in a per-ear centred frame so the layer sees SHAPE, not absolute
        # position -- ear location in the scan is irrelevant to curve geometry.
        centre = anchors.mean(dim=1, keepdim=True)
        x = anchors - centre

        if self.use_confidence:
            if confidence is None:
                confidence = torch.zeros(B, anchors.shape[1], 1, device=anchors.device)
            x = torch.cat([x, confidence], dim=2)

        h = self.input_proj(x) + self.slot_embed.weight.unsqueeze(0)

        out = torch.zeros_like(h)
        for slots in CONTOUR_ANCHOR_SLOTS.values():
            idx = torch.tensor(slots, device=anchors.device, dtype=torch.long)
            # attention WITHIN this contour only
            out[:, idx, :] = self.encoder(h[:, idx, :])

        return anchors + self.out(out)


def anchors_to_tensor(corrected: dict) -> torch.Tensor:
    """{anchor_idx: xyz} -> (1, 15, 3) in ANCHOR_INDICES order."""
    return torch.tensor(
        np.array([corrected[a] for a in ANCHOR_INDICES], dtype=np.float32)).unsqueeze(0)


def tensor_to_anchors(t: torch.Tensor) -> dict:
    """(1, 15, 3) -> {anchor_idx: xyz}."""
    arr = t.detach().squeeze(0).cpu().numpy()
    return {a: arr[i] for i, a in enumerate(ANCHOR_INDICES)}


# ------------------------------------------------------- training data (path b)

def generate_curve_examples(dataset, subject_ids, k_inner: int = 4,
                             k_references: int = 7, seed: int = 0,
                             capacity: str = "large", n_epochs: int = 150,
                             heatmap: bool = False, heatmap_sigma: float = 1.5,
                             torch_seed: int | None = None, verbose: bool = True,
                             **patch_kw):
    """(predicted 15 anchors, true 15 anchors) pairs with a REALISTIC error
    distribution.

    Why the nested structure is necessary. The obvious shortcut -- run the
    trained anchor model over its own training subjects -- produces predictions
    that are far too good: measured training error is 0.495-0.590mm against
    ~1.66mm on held-out ears, because the model has partly memorised those
    subjects. generate_polish_examples accepts that compromise, and can afford
    to: it corrects small residuals, which still exist on memorised data.

    The curve layer cannot afford it. Its entire purpose is repairing WRONG-RIDGE
    errors -- an anchor landing on the inner fold instead of the rim -- and those
    are precisely the large errors a model does not make on data it memorised.
    Trained on the shortcut, it would essentially never see the failure it
    exists to fix.

    So: split `subject_ids` into k_inner folds and, for each, train an anchor
    model on the OTHER folds and predict the held-out one. Every prediction then
    comes from a model that never saw that subject.

    Single models (not ensembles) are used for this generation pass -- ~4 extra
    trainings per outer fold rather than 4 x n_seeds. The resulting errors are
    slightly LARGER than an ensemble's, which is the safe direction: the layer
    sees harder cases than it will face.
    """
    from ..correction.anchor_model import train_model
    from ..foundations.canonical import iter_landmarks_only, load_canonical_mesh
    from ..foundations.contours import ANCHOR_POSITION
    from ..correction.heatmap_model import predict_any_correction, train_heatmap_model
    from ..correction.patch_features import crop_submesh, extract_patch
    from ..registration.registration import compute_raw_full
    from ..foundations.splits import k_fold_subject_split
    from ..registration.template import build_template
    from ..correction.training_data import generate_inner_cv_examples

    examples = []
    for fold_i, (tr_ids, val_ids) in enumerate(k_fold_subject_split(subject_ids, k_inner, seed)):
        if verbose:
            print(f"  curve fold {fold_i + 1}/{k_inner}: train {len(tr_ids)} -> predict {len(val_ids)}")

        inner_ex = generate_inner_cv_examples(dataset, tr_ids, k_inner=max(2, k_inner - 1),
                                               k_references=k_references, seed=seed + fold_i,
                                               verbose=False, **patch_kw)
        ts = None if torch_seed is None else torch_seed + 500 + fold_i
        if heatmap:
            model = train_heatmap_model(inner_ex, n_epochs=n_epochs, capacity=capacity,
                                         sigma=heatmap_sigma, verbose=False, torch_seed=ts)
        else:
            model = train_model(inner_ex, n_epochs=n_epochs, capacity=capacity,
                                 verbose=False, torch_seed=ts)

        template = build_template(dataset, tr_ids, k_references=k_references)

        for ex in iter_landmarks_only(dataset, val_ids):
            mesh = load_canonical_mesh(dataset, ex.subject_id, ex.side)
            raw = compute_raw_full(template, mesh.vertices, method="tps")
            local = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)
            del mesh

            pred = []
            for a in ANCHOR_INDICES:
                patch = extract_patch(local, raw[a], **{k: v for k, v in patch_kw.items()
                                                         if k in ("gaussian", "geodesic")})
                pred.append(raw[a] + predict_any_correction(model, patch, ANCHOR_POSITION[a]))

            examples.append({
                "pred": np.array(pred, dtype=np.float32),
                "truth": ex.landmarks[ANCHOR_INDICES].astype(np.float32),
                "subject_id": ex.subject_id, "side": ex.side,
            })

    if verbose and examples:
        e = np.array([np.linalg.norm(x["pred"] - x["truth"], axis=1).mean() for x in examples])
        print(f"  curve examples: {len(examples)} ears, input anchor error "
              f"mean {e.mean():.3f}mm p95 {np.percentile(e, 95):.3f}mm "
              f"(should resemble TEST error, not training error)")
    return examples


def train_curve_layer(examples, n_epochs: int = 300, batch_size: int = 32,
                       lr: float = 3e-4, weight_decay: float = 1e-4,
                       hidden: int = HIDDEN, n_heads: int = N_HEADS,
                       n_layers: int = N_LAYERS, device=None,
                       torch_seed: int | None = None, verbose: bool = True):
    """Train the curve layer on (pred, truth) anchor pairs.

    AdamW at 3e-4 follows the teammate's S111 surface-query spec, which is a
    validated operating point for a transformer of this size on this problem.
    """
    from ..correction.anchor_model import TRAIN_DEVICE
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    device = device or TRAIN_DEVICE

    P = torch.from_numpy(np.stack([e["pred"] for e in examples]))
    T = torch.from_numpy(np.stack([e["truth"] for e in examples]))
    ds = torch.utils.data.TensorDataset(P, T)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = CurveLayer(hidden=hidden, n_heads=n_heads, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    baseline = torch.linalg.norm(P - T, dim=2).mean().item()
    for epoch in range(n_epochs):
        model.train()
        tot = 0.0
        for p, t in loader:
            p, t = p.to(device), t.to(device)
            opt.zero_grad()
            loss = torch.linalg.norm(model(p) - t, dim=2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(p)
        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d}  train_anchor_err={tot / len(examples):.4f}mm "
                  f"(input was {baseline:.4f}mm)")
    return model.to("cpu").eval()


@torch.no_grad()
def apply_curve_layer(model, corrected: dict) -> dict:
    """{anchor_idx: xyz} -> adjusted {anchor_idx: xyz}."""
    return tensor_to_anchors(model(anchors_to_tensor(corrected)))
