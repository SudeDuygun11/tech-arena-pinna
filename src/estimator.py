"""Challenge submission entry point.

Loads a saved bundle (template + anchor ensemble + polish ensemble) and runs the
full documented pipeline. See src/pipeline.py for the stage-by-stage reasoning.

WHY THIS FILE WAS REWRITTEN
---------------------------
The previous version could not instantiate the evaluated model. It:
  * constructed `PatchCorrector()` with DEFAULT arguments -- the `small` preset,
    22,387 params -- while every reported result used `large` (87,779);
  * loaded a SINGLE model, while the evaluated configuration is a 3-5 member
    multi-seed ensemble;
  * never loaded or passed a polish model, silently skipping Stage 2b;
  * pointed at an `assets/` directory that was never created, so it could not
    even construct.

The effect was that the submission path corresponded to roughly the 2.87mm
stage of the error table, not the documented 1.5931mm -- and in practice raised
FileNotFoundError. This version reads the architecture from the saved bundle
instead of hardcoding defaults, so it cannot silently disagree with training.

Produce a bundle with:
    python scripts/cross_validate.py ... --save-models assets/bundle.pt


ROLE IN THE PIPELINE
--------------------
Reading order : 15 of 20   (src root)
Duty          : Submission entry point.

Loads a saved bundle (template, anchor ensemble, polish ensemble) and
predicts both ears. Reads the architecture FROM the bundle rather than
hardcoding it, so it cannot silently disagree with how the models were
trained.

Known issues / status:
  Shows zero importers because the challenge harness calls it, not our own
  code. Do not delete it as dead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from trimesh import Trimesh

from .correction.anchor_model import PatchCorrector
from .foundations.contours import N_ANCHORS, N_LANDMARKS
from .foundations.geometry import mirror_y
from .correction.heatmap_model import HeatmapCorrector
from .pipeline import predict_canonical
from .registration.registration import ReferenceTemplate, Template

DEFAULT_BUNDLE = Path(__file__).resolve().parent.parent / "assets" / "bundle.pt"


def _build(spec: dict, n_classes: int):
    """Reconstruct one model from its saved architecture spec."""
    kind = spec.get("kind", "patch")
    cap = spec["capacity"]
    feat = spec.get("point_feature_dim", 7)
    if kind == "heatmap":
        m = HeatmapCorrector(n_anchor_classes=n_classes, point_feature_dim=feat, capacity=cap)
    else:
        m = PatchCorrector.from_capacity(cap, n_anchor_classes=n_classes,
                                          pooling=spec.get("pooling", "max"),
                                          point_feature_dim=feat, ctx_dim=spec.get("ctx_dim", 0))
    m.load_state_dict(spec["state_dict"])
    return m.eval()


class LandmarkExtractor:
    """Full pipeline: registration -> anchor ensemble -> blend -> polish ensemble."""

    def __init__(self, bundle_path: str = str(DEFAULT_BUNDLE)):
        path = Path(bundle_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No model bundle at {path}. Create one with:\n"
                f"  python scripts/cross_validate.py --data-dir <...> "
                f"--with-correction --with-polish --capacity large "
                f"--multi-seed-ensemble --n-seeds 3 --save-models {path}"
            )
        # weights_only=False is required for the template dataclasses; only load
        # bundles you produced yourself.
        b = torch.load(path, map_location="cpu", weights_only=False)

        self.template: Template = b["template"]
        self.anchor_models = [_build(s, N_ANCHORS) for s in b["anchor_models"]]
        self.polish_models = [_build(s, N_LANDMARKS) for s in b.get("polish_models", [])]
        self.config = b.get("config", {})

        # inference-time settings must match how the models were trained
        self.gaussian = self.config.get("gaussian_curvature", False)
        # channels zeroed during training must be zeroed at inference too, or the
        # model sees inputs it never saw in training
        from .correction.patch_features import set_dropped_features
        set_dropped_features(self.config.get("drop_features", []))
        from .correction.patch_features import set_point_features
        set_point_features(self.config.get("point_features", "base"))
        from .registration.registration import set_reference_selection
        set_reference_selection(self.config.get("select_references"))
        from .registration.refbank import set_context_features
        set_context_features(self.config.get("context_features", False))
        from .correction.alignment import set_aligned_patches
        set_aligned_patches(self.config.get("aligned_patches", False))
        self.geodesic = self.config.get("geodesic_patch", False)
        self.patch_points = self.config.get("patch_points")
        self.patch_radius = self.config.get("patch_radius")
        self.registration_method = self.config.get("registration_method", "tps")

    def describe(self) -> str:
        return (f"{len(self.anchor_models)} anchor + {len(self.polish_models)} polish model(s), "
                f"capacity={self.config.get('capacity')}, "
                f"k_references={len(self.template.references)}, "
                f"registration={self.registration_method}")

    def extract(self, mesh: Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (left_landmarks, right_landmarks), each (85, 3), in the
        ORIGINAL mesh frame."""
        anchors = self.anchor_models if len(self.anchor_models) > 1 else self.anchor_models[0]
        polish = (self.polish_models if len(self.polish_models) > 1
                  else (self.polish_models[0] if self.polish_models else None))

        kw = dict(polish_model=polish, registration_method=self.registration_method,
                  gaussian=self.gaussian, geodesic=self.geodesic,
                  patch_points=self.patch_points, patch_radius=self.patch_radius)

        left = predict_canonical(self.template, anchors, mesh, **kw)

        mirrored = mesh.copy()
        mirrored.vertices = mirror_y(mesh.vertices)
        # Reverse face winding as well -- mirror_y is a reflection (det = -1), so
        # leaving the winding alone makes every computed vertex normal point
        # INWARD, which flips 3 of the 7 patch features and the sign of the
        # curvature feature derived from them. Must match canonical.py exactly:
        # if training and inference disagree on this, right ears are evaluated
        # with a convention the model never saw.
        mirrored.faces = np.asarray(mirrored.faces)[:, ::-1]
        right = mirror_y(predict_canonical(self.template, anchors, mirrored, **kw))

        return left, right
