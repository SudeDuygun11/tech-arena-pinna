"""Nested subject-wise cross-validation of the full pipeline against the
official challenge metric.

Outer folds hold out subjects entirely for scoring. Within each outer
training pool, an inner k-fold generates leakage-free training examples for
the Stage-1 correction network (the registration template for each inner
fold is built only from the OTHER inner folds).

Usage:
    python scripts/cross_validate.py --data-dir "<mesh/landmarks parent>" \
        --n-subjects 100 --outer-folds 3 --inner-folds 4 --k-references 5 \
        --epochs 30 --seed 0 --with-correction
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT_DIR = Path(__file__).resolve().parent.parent

from src.correction.anchor_model import train_model, train_snapshot_ensemble, train_multi_seed_ensemble
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import CONTOUR_SPECS, ANCHOR_INDICES, N_LANDMARKS
from src.foundations.dataset import Dataset
from src.metrics import compute_mean_landmark_distance
from src.pipeline import predict_canonical
from src.foundations.splits import k_fold_subject_split, single_subject_split
from src.experimental.curve_layer import generate_curve_examples, train_curve_layer
from src.correction.heatmap_model import train_heatmap_ensemble, train_heatmap_model
from src.experimental.joint_fit import calibrate_confidence, fit_shape_prior
from src.registration.template import build_template
from src.correction.training_data import (AUGMENT_JITTER_STD, assign_contour_weights, boost_hard_examples,
                                 boost_landmark_weight,
                                 generate_inner_cv_examples, generate_polish_examples,
                                 train_per_contour_anchor_models)


def evaluate_pipeline(dataset, template, model, subject_ids, polish_model=None,
                       registration_method="tps", tta_views=1, multiscale=False, gaussian=False,
                       crest=False, shape_prior=None, prior_strength=0.05,
                       members_per_point=None, geodesic=False, patch_points=None,
                       patch_radius=None, curve_model=None,
                       equidistant=True, derive_terminals=False, two_pass=False):
    per_point_errors = []  # list of (85,) arrays, ear-level
    rows = []
    for tid in subject_ids:
        for side in ["left", "right"]:
            ex = [e for e in iter_landmarks_only(dataset, [tid]) if e.side == side][0]
            mesh = load_canonical_mesh(dataset, tid, side)
            pred_canonical = predict_canonical(template, model, mesh, polish_model=polish_model,
                                                registration_method=registration_method,
                                                tta_views=tta_views, multiscale=multiscale,
                                                gaussian=gaussian, crest=crest,
                                                shape_prior=shape_prior,
                                                prior_strength=prior_strength,
                                                members_per_point=members_per_point,
                                                geodesic=geodesic, patch_points=patch_points,
                                                patch_radius=patch_radius,
                                                curve_model=curve_model,
                                                equidistant=equidistant,
                                                derive_terminals=derive_terminals,
                                                two_pass=two_pass)
            gt_canonical = ex.landmarks  # already canonical (mirrored for 'right' upstream)

            dist = np.linalg.norm(pred_canonical - gt_canonical, axis=1)
            per_point_errors.append(dist)
            rows.append({"subject_id": tid, "side": side, "mean_dist": dist.mean(),
                          "pred": pred_canonical, "truth": gt_canonical})
    return np.array(per_point_errors), rows


def summarize(per_point_errors: np.ndarray, label: str):
    overall = per_point_errors.mean()
    print(f"[{label}] official mean landmark distance: {overall:.4f}  "
          f"(n_ears={len(per_point_errors)})")
    for name, spec in CONTOUR_SPECS.items():
        s, e = spec["range"]
        print(f"    {name:20s} mean={per_point_errors[:, s:e].mean():.4f}")
    anchor_err = per_point_errors[:, ANCHOR_INDICES].mean()
    print(f"    {'anchors only':20s} mean={anchor_err:.4f}")
    return overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="parent of mesh/ and landmarks/")
    p.add_argument("--n-subjects", type=int, default=100)
    p.add_argument("--outer-folds", type=int, default=3)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--k-references", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0,
                    help="numpy seed: subject splits and augmentation.")
    p.add_argument("--torch-seed", type=int, default=None,
                    help="seed for torch weight-init/shuffling. Leave unset for the old "
                         "(unseeded, non-reproducible) behaviour; set it to make a run "
                         "reproducible, and VARY it across repeats to measure training "
                         "variance -- an A/B difference smaller than that variance is "
                         "not interpretable.")
    p.add_argument("--with-correction", action="store_true",
                    help="train and apply the Stage-1 learned correction network; "
                         "otherwise evaluate the registration-only baseline")
    p.add_argument("--with-polish", action="store_true",
                    help="also train and apply the Stage-2 per-point polish network "
                         "(requires --with-correction)")
    p.add_argument("--polish-epochs", type=int, default=None,
                    help="defaults to --epochs if not set")
    p.add_argument("--per-contour", action="store_true",
                    help="train 4 independent anchor-correction networks (one per contour) "
                         "instead of 1 shared network -- tests whether per-contour "
                         "specialization beats shared-weight data efficiency. "
                         "Requires --with-correction; not compatible with --with-polish.")
    p.add_argument("--registration-method", choices=["tps", "cpd", "hybrid"], default="tps",
                    help="Stage-1a registration variant. 'hybrid' is much slower (~25x) since "
                         "it runs both TPS and CPD in full every call.")
    p.add_argument("--loss", choices=["smooth_l1", "wing"], default="smooth_l1",
                    help="loss function for the anchor/polish correction networks")
    p.add_argument("--snapshot-ensemble", action="store_true",
                    help="train via cosine-annealed cyclic LR, saving one snapshot per cycle, "
                         "and average predictions across snapshots at inference "
                         "('M models for the training cost of 1'). Not compatible with --per-contour.")
    p.add_argument("--n-cycles", type=int, default=4)
    p.add_argument("--epochs-per-cycle", type=int, default=None,
                    help="defaults to --epochs / --n-cycles if not set")
    p.add_argument("--jitter-std", type=float, default=None,
                    help="anchor-stage augmentation jitter std, mm (default: "
                         "training_data.AUGMENT_JITTER_STD, currently 1.5mm)")
    p.add_argument("--capacity", choices=["small", "large", "xlarge", "xxlarge"], default="small",
                    help="network capacity preset. 'large' is ~4x parameters "
                         "(scoped ablation for whether more capacity helps now that "
                         "training uses the full subject pool, not a data-starved subset).")
    p.add_argument("--multi-seed-ensemble", action="store_true",
                    help="train N fully independent models (distinct init/shuffling) and "
                         "average their predictions -- more diverse than --snapshot-ensemble, "
                         "at N x training cost. Not compatible with --snapshot-ensemble/--per-contour.")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--tta-views", type=int, default=1,
                    help="test-time augmentation: average predictions over N randomly-resampled "
                         "patch views per point at inference (1 = off).")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                    help="Adam weight decay -- larger capacity models likely need more "
                         "regularization than the small-capacity default was tuned for.")
    p.add_argument("--contour-weighted-loss", action="store_true",
                    help="upweight persistently-harder contours (inner_helix, outer_helix) and "
                         "downweight the easiest (concha_outline) in the training loss -- see "
                         "training_data.DEFAULT_CONTOUR_LOSS_WEIGHTS.")
    p.add_argument("--geodesic-patch", action="store_true",
                    help="build patches from an ALONG-SURFACE (geodesic) neighbourhood instead "
                         "of a Euclidean ball. Measured on real ears: 33.5%% of Euclidean-ball "
                         "vertices are across a fold (54.6%% for superior_antihelix) -- "
                         "Euclidean-near but surface-far geometry the network cannot "
                         "distinguish from the ridge it should be reading.")
    p.add_argument("--save-models", default=None, metavar="PATH",
                    help="save a deployable bundle (template + anchor ensemble + polish "
                         "ensemble + config) from the LAST outer fold, for use by "
                         "src/estimator.LandmarkExtractor. Without this no checkpoints "
                         "exist at all and no submission can be produced.")
    p.add_argument("--curve-layer", action="store_true",
                    help="after per-anchor correction, let anchors on the same contour "
                         "attend to each other before interpolation. Anchors sit where the "
                         "curve TURNS (measured: 1.34-2.20x sharper than neighbouring "
                         "points), which is a property of the CURVE that a 7mm patch cannot "
                         "see. Training data is generated nested (see "
                         "curve_layer.generate_curve_examples) so the layer sees realistic "
                         "wrong-ridge errors. See src/curve_layer.py.")
    p.add_argument("--curve-epochs", type=int, default=300)
    p.add_argument("--no-equidistant", dest="equidistant", action="store_false",
                    help="disable uniform arc-length re-spacing of interpolated points. "
                         "The ground truth was BUILT by uniform redistribution between the "
                         "annotator's fixed points; enforcing it measured -9.7%% (1.5326 -> "
                         "1.3837) on 400 ears. On by default; this flag is for ablation.")
    p.add_argument("--two-pass-registration", dest="two_pass", action="store_true",
                    help="run registration TWICE: once blind for rough anchors, then again "
                         "with those anchors pinned as hard rigid-ICP correspondences. "
                         "Measured registration-only (2026-09-10): -0.43mm on the 70 "
                         "non-anchor points, robust to 0-3mm of anchor error. Roughly "
                         "doubles registration cost per ear. See registration._rigid_icp.")
    p.add_argument("--derive-terminals", action="store_true",
                    help="derive landmark 74 by stepping one gap past 73 along the "
                         "registration prior's path, instead of using the model's "
                         "prediction for it. 74 is defined by arithmetic, not anatomy, and "
                         "is the worst landmark in the set (3.451mm).")
    p.add_argument("--heatmap", action="store_true",
                    help="use dense per-point HEATMAP prediction for the anchor stage "
                         "instead of direct coordinate regression: score all 256 patch "
                         "points, soft-argmax over their positions, plus a residual offset "
                         "head to escape the patch hull (~10%% of anchors lie outside a 7mm "
                         "patch). This is the dominant paradigm in the landmark-localization "
                         "literature. See src/heatmap_model.py.")
    p.add_argument("--heatmap-sigma", type=float, default=1.5,
                    help="mm; width of the Gaussian heatmap target (default 1.5 -> ~48 "
                         "effective supervising points per example, measured on real patches)")
    p.add_argument("--hard-example-mining", action="store_true",
                    help="train once, upweight the worst-predicted training examples, then "
                         "RETRAIN (doubles training cost). Motivated by our own measurement "
                         "that REMOVING the noisiest 15%% of examples cost +7.1%% -- if those "
                         "were label noise, dropping them would have helped, so they carry "
                         "signal. See training_data.boost_hard_examples.")
    p.add_argument("--hard-fraction", type=float, default=0.25,
                    help="share of examples treated as hard (default 0.25)")
    p.add_argument("--hard-boost", type=float, default=2.0,
                    help="loss-weight multiplier for hard examples (default 2.0)")
    p.add_argument("--patch-radius", type=float, default=None,
                    help="patch radius in mm (default 7.0). Larger gives more context per "
                         "point at lower sampling density.")
    p.add_argument("--patch-points", type=int, default=None,
                    help="points sampled per patch (default 256). A 7mm patch holds ~450 "
                         "vertices, so the default discards nearly half the available geometry.")
    p.add_argument("--members-per-point", type=int, default=None,
                    help="assign a rotating subset of ensemble members to each landmark "
                         "instead of averaging all of them. Decorrelates neighbouring "
                         "landmarks' errors, which is what decides whether a shape prior "
                         "can denoise them. Costs ~5.5%% per-point accuracy; the prior is "
                         "meant to more than repay it. See src/pipeline._member_subset.")
    p.add_argument("--save-predictions", default=None, metavar="PATH",
                    help="write predicted and ground-truth landmarks to an .npz. Needed to "
                         "measure the SPATIAL CORRELATION of our errors, which determines "
                         "whether a shape prior can denoise them: independent error is "
                         "filtered well (1.48->0.71mm) and fully-coherent drift is absorbed "
                         "by Procrustes, but 5-10mm correlated error passes straight through "
                         "(1.48->1.13mm). See scripts/correlated_noise_test.py.")
    p.add_argument("--shape-prior", action="store_true",
                    help="constrain the 15 corrected anchors with a PCA shape prior fitted "
                         "on TRAINING subjects only (85 points, 45 modes, strength 0.05, "
                         "E-CPV confidence weighting). Requires --multi-seed-ensemble: with "
                         "uniform weights the prior measured WORSE than no prior. Reports "
                         "BOTH arms as a PAIRED comparison -- same models, same ears -- so "
                         "training noise cancels. See src/joint_fit.py.")
    p.add_argument("--prior-strength", type=float, default=0.05)
    p.add_argument("--prior-modes", type=int, default=45)
    p.add_argument("--crest-attract", action="store_true",
                    help="pull interpolated (non-anchor) points of outer_helix/inner_helix/"
                         "concha_outline partway toward nearby high-curvature ridge vertices. "
                         "superior_antihelix is excluded -- its measured crest signal is no "
                         "better than our existing error. See interpolation.crest_attract.")
    p.add_argument("--drop-features", nargs="+", default=[],
                   choices=["xyz", "normal", "curv", "gauss"],
                   help="zero these patch input channels (per-landmark feature study; "
                        "shapes unchanged, recorded in the saved bundle)")
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam",
                   help="adamw = decoupled weight decay (use --weight-decay ~1e-2)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss-weighting", choices=["none", "uncertainty"], default="none",
                   help="uncertainty = learned per-anchor-class loss weights (Kendall et al. 2018)")
    p.add_argument("--augment-rotate", type=float, default=0.0, help="max random rotation (deg)")
    p.add_argument("--augment-scale", type=float, default=0.0, help="random scale +- fraction")
    p.add_argument("--augment-dropout", type=float, default=0.0,
                   help="drop up to this fraction of patch points per example")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--swa-start", type=float, default=0.0,
                   help="average weights from this fraction of training on (0 = off)")
    p.add_argument("--wing-omega", type=float, default=3.0)
    p.add_argument("--wing-epsilon", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-subject-fraction", type=float, default=1.0, metavar="F",
                   help="learning-curve test: train the anchor network on a random F of the "
                        "outer-training SUBJECTS (both ears kept together), subset seeded by "
                        "--torch-seed; the registration template still uses all of them")
    p.add_argument("--polish-legacy", action="store_true",
                   help="train the polish networks with the pre-recipe settings (SmoothL1, no "
                        "augmentation/SWA); the anchor network still uses the flags above")
    p.add_argument("--aligned-patches", action="store_true",
                   help="express each anchor patch and its correction in a local contour/normal frame")
    p.add_argument("--context-features", action="store_true",
                   help="feed the anchor network reference-agreement context (needs --select-references)")
    p.add_argument("--select-references", type=int, default=None, metavar="K",
                   help="per ear, average only the K best-fitting of the --k-references pool")
    p.add_argument("--point-features", choices=["base", "curvedness", "rich"], default="base",
                   help="per-point patch channels (see patch_features.extract_patch)")
    p.add_argument("--example-cache", default=None, metavar="DIR",
                   help="cache generated anchor training examples per fold+setting (screening)")
    p.add_argument("--only-fold", type=int, default=None,
                   help="run just this outer fold index (0-based) -- for screening "
                        "a variant against the same fold of a baseline")
    p.add_argument("--gaussian-curvature", action="store_true",
                    help="append discrete Gaussian curvature (angle deficit) as an 8th per-point "
                         "feature -- distinguishes saddles from domes/bowls, which the existing "
                         "mean-curvature-like proxy cannot.")
    p.add_argument("--multiscale-patch", action="store_true",
                    help="split the 256-point patch budget across a fine (7mm) and coarse (14mm) "
                         "radius instead of one fixed radius, giving both local ridge detail and "
                         "broader contour context (src.patch_features.extract_multiscale_patch).")
    p.add_argument("--pooling", choices=["max", "max_mean", "attention"], default="max",
                    help="point-cloud aggregation: 'max' (current default, discards a lot of "
                         "information), 'max_mean' (concatenate max+mean pooling), or "
                         "'attention' (lightweight single-layer learned softmax-weighted pooling).")
    p.add_argument("--anchor74-boost", type=float, default=1.0,
                    help="loss-weight multiplier for landmark 74 (the 'derived continuation' "
                         "point, structurally different from the other 14 anchors) on the "
                         "anchor-stage training examples. 1.0 = no boost.")
    p.add_argument("--train-fraction", type=float, default=None,
                    help="use a single fixed train/test split at this ratio (e.g. 0.85 or 0.90) "
                         "instead of k-fold CV via --outer-folds. One split, not repeated.")
    args = p.parse_args()
    if args.per_contour:
        assert args.with_correction, "--per-contour requires --with-correction"
        assert not args.with_polish, "--per-contour + --with-polish not supported yet"

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                 landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    subject_ids = ds.subject_ids[:args.n_subjects]
    jitter = args.jitter_std if args.jitter_std is not None else AUGMENT_JITTER_STD
    if args.train_fraction is not None:
        split_iter = list(single_subject_split(subject_ids, args.train_fraction, args.seed))
        n_folds = 1
        print(f"Using {len(subject_ids)} subjects, single {args.train_fraction:.0%}/"
              f"{1-args.train_fraction:.0%} split, "
              f"with_correction={args.with_correction}, registration_method={args.registration_method}, "
              f"loss={args.loss}, jitter_std={jitter}")
    else:
        split_iter = list(k_fold_subject_split(subject_ids, args.outer_folds, args.seed))
        n_folds = args.outer_folds
        print(f"Using {len(subject_ids)} subjects, outer_folds={args.outer_folds}, "
              f"with_correction={args.with_correction}, registration_method={args.registration_method}, "
              f"loss={args.loss}, jitter_std={jitter}")

    from src.correction.patch_features import set_dropped_features
    set_dropped_features(args.drop_features)
    from src.registration.registration import set_reference_selection
    set_reference_selection(args.select_references)
    from src.registration.refbank import set_context_features
    if args.context_features:
        assert args.select_references, "--context-features requires --select-references"
        assert args.registration_method == "tps", "context features are implemented for TPS only"
    set_context_features(args.context_features)
    from src.correction.alignment import set_aligned_patches
    if args.aligned_patches:
        assert not (args.per_contour or args.heatmap or args.multiscale_patch), \
            "--aligned-patches supports the single shared regression network only"
    set_aligned_patches(args.aligned_patches)
    if args.select_references:
        print(f"per-ear reference selection: best {args.select_references} of "
              f"{args.k_references} references by surface fit")
    from src.correction.patch_features import set_point_features, FEATURE_VERSION
    set_point_features(args.point_features)
    opt_kw = dict(optimizer=args.optimizer, lr=args.lr, batch_size=args.batch_size)
    # regression trainers only (the heatmap trainer has its own loss and no augmentation)
    reg_kw = dict(opt_kw, loss_weighting=args.loss_weighting, augment_rotate=args.augment_rotate,
                  augment_scale=args.augment_scale, augment_dropout=args.augment_dropout,
                  lr_schedule=args.lr_schedule, swa_start=args.swa_start,
                  wing_omega=args.wing_omega, wing_epsilon=args.wing_epsilon)
    print(f"point features: {args.point_features}; optimizer {args.optimizer} lr {args.lr} "
          f"batch {args.batch_size} wd {args.weight_decay}")
    if args.drop_features:
        print(f"patch channels ZEROED for this run: {args.drop_features}")

    all_errors = []
    prior_errors = []
    saved_rows = []
    for fold_i, (train_ids, test_ids) in enumerate(split_iter):
        if args.only_fold is not None and fold_i != args.only_fold:
            continue
        t0 = time.time()
        print(f"\n=== outer fold {fold_i + 1}/{n_folds}: "
              f"{len(train_ids)} train / {len(test_ids)} test ===")

        model = None
        if args.with_correction:
            print(" generating leakage-free inner-CV training examples...")
            gen_kw = dict(k_inner=args.inner_folds, k_references=args.k_references,
                          seed=args.seed, verbose=True,
                          registration_method=args.registration_method,
                          jitter_std=args.jitter_std if args.jitter_std is not None
                          else AUGMENT_JITTER_STD,
                          multiscale=args.multiscale_patch, gaussian=args.gaussian_curvature,
                          crest=args.crest_attract,
                          # previously NOT passed although evaluation used them:
                          # training and test patches differed whenever set
                          geodesic=args.geodesic_patch, patch_points=args.patch_points,
                          patch_radius=args.patch_radius)
            cache_file = None
            if args.example_cache:
                import hashlib, pickle
                # the sources of registration and patch extraction are part of the key, so
                # any code change there regenerates examples instead of reusing stale ones
                src_sig = hashlib.sha1(b"".join(
                    (ROOT_DIR / f).read_bytes() for f in ("src/registration/registration.py",
                                                          "src/registration/refbank.py",
                                                          "src/correction/patch_features.py",
                                                          "src/correction/alignment.py",
                                                          "src/correction/training_data.py"))).hexdigest()
                key = repr((fold_i, sorted(train_ids), sorted((k, str(v)) for k, v in gen_kw.items()
                                                              if k != "verbose"),
                            FEATURE_VERSION, args.point_features, sorted(args.drop_features),
                            args.select_references, args.context_features, args.aligned_patches, src_sig))
                cache_file = Path(args.example_cache) / f"examples_{hashlib.sha1(key.encode()).hexdigest()[:16]}.pkl"
            if cache_file is not None and cache_file.exists():
                import pickle
                examples = pickle.loads(cache_file.read_bytes())
                print(f" loaded {len(examples)} cached examples from {cache_file.name}")
            else:
                examples = generate_inner_cv_examples(ds, train_ids, **gen_kw)
                if cache_file is not None:
                    import pickle
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_bytes(pickle.dumps(examples))
                    print(f" cached {len(examples)} examples -> {cache_file.name}")
            if args.train_subject_fraction < 1.0:
                order = list(train_ids)
                np.random.default_rng(1000 + args.torch_seed).shuffle(order)
                keep = set(order[:max(1, int(round(args.train_subject_fraction * len(order))))])
                n_before = len(examples)
                examples = [e for e in examples if e["subject_id"] in keep]
                print(f" learning-curve subset: {len(keep)}/{len(order)} subjects, "
                      f"{len(examples)}/{n_before} examples")
            if args.contour_weighted_loss:
                examples = assign_contour_weights(examples)
                print(" applied per-contour loss weights to anchor examples")
            if args.anchor74_boost != 1.0:
                examples = boost_landmark_weight(examples, 74, args.anchor74_boost)
                print(f" applied {args.anchor74_boost}x loss-weight boost to anchor 74")
            if args.per_contour:
                print(f" training 4 per-contour correction networks on "
                      f"{len(examples)} total examples...")
                model = train_per_contour_anchor_models(examples, n_epochs=args.epochs,
                                                         capacity=args.capacity, loss_type=args.loss,
                                                         weight_decay=args.weight_decay,
                                                         torch_seed=args.torch_seed,
                                                         pooling=args.pooling, **reg_kw)
            elif args.snapshot_ensemble:
                epc = args.epochs_per_cycle or max(1, args.epochs // args.n_cycles)
                print(f" training snapshot ensemble ({args.n_cycles} cycles x {epc} epochs) on "
                      f"{len(examples)} examples (loss={args.loss})...")
                model = train_snapshot_ensemble(examples, n_cycles=args.n_cycles,
                                                 epochs_per_cycle=epc, loss_type=args.loss,
                                                 capacity=args.capacity)
            elif args.multi_seed_ensemble:
                print(f" training multi-seed ensemble ({args.n_seeds} independent models) on "
                      f"{len(examples)} examples (loss={args.loss})...")
                if args.heatmap:
                    print(f" training HEATMAP ensemble ({args.n_seeds} models, "
                          f"sigma={args.heatmap_sigma}mm)...")
                    model = train_heatmap_ensemble(examples, n_models=args.n_seeds,
                                                    n_epochs=args.epochs,
                                                    capacity=args.capacity,
                                                    weight_decay=args.weight_decay,
                                                    sigma=args.heatmap_sigma, verbose=True,
                                                    base_seed=args.torch_seed, **opt_kw)
                else:
                    model = train_multi_seed_ensemble(examples, n_models=args.n_seeds,
                                                       base_seed=args.torch_seed,
                                                       outer_fold=fold_i, stage=0, **reg_kw,
                                                       n_epochs=args.epochs, loss_type=args.loss,
                                                       capacity=args.capacity,
                                                       weight_decay=args.weight_decay)
            else:
                print(f" training shared correction network on {len(examples)} examples "
                      f"(loss={args.loss}, capacity={args.capacity}, wd={args.weight_decay})...")
                model = train_model(examples, n_epochs=args.epochs, loss_type=args.loss,
                                     capacity=args.capacity, weight_decay=args.weight_decay,
                                     pooling=args.pooling, torch_seed=args.torch_seed, **reg_kw)

        if args.hard_example_mining and model is not None:
            print(f" hard-example mining: re-weighting worst {args.hard_fraction:.0%} "
                  f"x{args.hard_boost} and retraining...")
            examples = boost_hard_examples(examples, model, args.hard_fraction, args.hard_boost)
            if args.multi_seed_ensemble:
                # NOTE: deliberately NOT passing torch_seed here -- it forwards to
                # every member, which would give all N identical weights and
                # silently collapse the ensemble. Members must differ.
                model = train_multi_seed_ensemble(examples, n_models=args.n_seeds,
                                                   base_seed=args.torch_seed,
                                                   outer_fold=fold_i, stage=2, **reg_kw,
                                                   n_epochs=args.epochs, loss_type=args.loss,
                                                   capacity=args.capacity,
                                                   weight_decay=args.weight_decay)
            else:
                model = train_model(examples, n_epochs=args.epochs, loss_type=args.loss,
                                     capacity=args.capacity, weight_decay=args.weight_decay,
                                     pooling=args.pooling, torch_seed=args.torch_seed, **reg_kw)

        curve_model = None
        if args.curve_layer and model is not None:
            print(" generating curve-layer examples (nested, leakage-free)...")
            cex = generate_curve_examples(
                ds, train_ids, k_inner=args.inner_folds, k_references=args.k_references,
                seed=args.seed, capacity=args.capacity, n_epochs=args.epochs,
                heatmap=args.heatmap, heatmap_sigma=args.heatmap_sigma,
                torch_seed=args.torch_seed, gaussian=args.gaussian_curvature,
                geodesic=args.geodesic_patch)
            print(f" training curve layer on {len(cex)} ears...")
            curve_model = train_curve_layer(
                cex, n_epochs=args.curve_epochs,
                torch_seed=(None if args.torch_seed is None else args.torch_seed + 900 + fold_i))

        print(" building outer-fold template from all training subjects...")
        template = build_template(ds, train_ids, k_references=args.k_references)

        polish_model = None
        if args.with_polish:
            assert model is not None, "--with-polish requires --with-correction"
            pol_kw = dict(opt_kw) if args.polish_legacy else reg_kw
            pol_loss = "smooth_l1" if args.polish_legacy else args.loss
            if args.polish_legacy:
                print(" polish networks: legacy training (SmoothL1, no augmentation, no SWA)")
            print(" generating polish-network training examples "
                  "(all 85 points, using the just-trained anchor model + outer template)...")
            polish_examples = generate_polish_examples(ds, template, model, train_ids, seed=args.seed,
                                                        registration_method=args.registration_method,
                                                        multiscale=args.multiscale_patch,
                                                        gaussian=args.gaussian_curvature,
                                                        crest=args.crest_attract,
                                                        geodesic=args.geodesic_patch,
                                                        patch_points=args.patch_points,
                                                        patch_radius=args.patch_radius)
            if args.contour_weighted_loss:
                polish_examples = assign_contour_weights(polish_examples)
                print(" applied per-contour loss weights to polish examples")
            if args.snapshot_ensemble:
                epc = args.epochs_per_cycle or max(1, (args.polish_epochs or args.epochs) // args.n_cycles)
                print(f" training polish snapshot ensemble ({args.n_cycles} cycles x {epc} epochs) on "
                      f"{len(polish_examples)} examples (loss={args.loss})...")
                polish_model = train_snapshot_ensemble(polish_examples, n_cycles=args.n_cycles,
                                                         epochs_per_cycle=epc, n_classes=N_LANDMARKS,
                                                         loss_type=args.loss, capacity=args.capacity)
            elif args.multi_seed_ensemble:
                print(f" training polish multi-seed ensemble ({args.n_seeds} independent models) on "
                      f"{len(polish_examples)} examples (loss={pol_loss})...")
                polish_model = train_multi_seed_ensemble(polish_examples, n_models=args.n_seeds,
                                                          base_seed=args.torch_seed,
                                                          outer_fold=fold_i, stage=1, **pol_kw,
                                                          n_epochs=args.polish_epochs or args.epochs,
                                                          n_classes=N_LANDMARKS, loss_type=pol_loss,
                                                          capacity=args.capacity,
                                                          weight_decay=args.weight_decay)
            else:
                print(f" training polish network on {len(polish_examples)} examples "
                      f"(loss={args.loss}, capacity={args.capacity}, wd={args.weight_decay})...")
                polish_model = train_model(polish_examples,
                                            n_epochs=args.polish_epochs or args.epochs,
                                            weight_decay=args.weight_decay,
                                            n_classes=N_LANDMARKS, loss_type=pol_loss,
                                            capacity=args.capacity, pooling=args.pooling, **pol_kw)

        prior = None
        if args.shape_prior:
            assert isinstance(model, list), \
                "--shape-prior requires --multi-seed-ensemble (E-CPV needs ensemble " \
                "members); with uniform weights the prior measured WORSE than none"
            # Fitted on TRAINING subjects' ground truth only -- never the scored ears.
            train_shapes = np.array([e.landmarks for e in iter_landmarks_only(ds, train_ids)])
            # Calibrate E-CPV -> error magnitude on the leakage-free inner-CV
            # examples, whose targets are the true residuals for those subjects.
            from src.correction.anchor_model import predict_correction
            cx, cy = [], []
            for e in examples:
                per = np.array([predict_correction(m, e["points"], e["anchor_class"])
                                for m in model])
                mu_d = per.mean(axis=0)
                cx.append(float(np.linalg.norm(per - mu_d, axis=1).mean()))
                cy.append(float(np.linalg.norm(mu_d - e["target"])))
            calib = calibrate_confidence(np.array(cx), np.array(cy))
            prior = fit_shape_prior(train_shapes, n_modes=args.prior_modes, calib=calib)
            print(f" shape prior: {len(train_shapes)} training shapes, {args.prior_modes} "
                  f"modes, calib err~{calib[0]:.3f}*ecpv+{calib[1]:.3f}")

        if args.save_models:
            import torch as _torch
            from pathlib import Path as _Path

            def _spec(m):
                kind = "heatmap" if type(m).__name__ == "HeatmapCorrector" else "patch"
                first = (m.point_mlp[0] if hasattr(m, "point_mlp") else None)
                return {"kind": kind, "capacity": args.capacity,
                        "ctx_dim": getattr(m, "ctx_dim", 0),
                        "point_feature_dim": (first.in_features if first is not None else 7),
                        "pooling": args.pooling, "state_dict": m.state_dict()}

            _am = model if isinstance(model, list) else [model]
            _pm = (polish_model if isinstance(polish_model, list)
                   else ([polish_model] if polish_model is not None else []))
            _out = _Path(args.save_models)
            if n_folds > 1:
                # one bundle per outer fold; previously every fold overwrote the same file
                _out = _out.with_name(f"{_out.stem}_fold{fold_i}{_out.suffix}")
            _out.parent.mkdir(parents=True, exist_ok=True)
            _torch.save({
                "template": template,
                "anchor_models": [_spec(m) for m in _am],
                "polish_models": [_spec(m) for m in _pm],
                "config": {
                    "capacity": args.capacity, "pooling": args.pooling,
                    "registration_method": args.registration_method,
                    "k_references": args.k_references,
                    "gaussian_curvature": args.gaussian_curvature,
                    "drop_features": list(args.drop_features),
                    "select_references": args.select_references,
                    "context_features": args.context_features,
                    "aligned_patches": args.aligned_patches,
                    "point_features": args.point_features, "optimizer": args.optimizer,
                    "lr": args.lr, "batch_size": args.batch_size,
                    "geodesic_patch": args.geodesic_patch,
                    "patch_points": args.patch_points, "patch_radius": args.patch_radius,
                    "heatmap": args.heatmap, "epochs": args.epochs,
                    "n_seeds": args.n_seeds if args.multi_seed_ensemble else 1,
                    "torch_seed": args.torch_seed, "outer_fold": fold_i,
                    "n_subjects": args.n_subjects,
                    "polish_legacy": args.polish_legacy,
                    "loss": args.loss, "augment_rotate": args.augment_rotate,
                    "augment_scale": args.augment_scale, "swa_start": args.swa_start,
                    "train_subject_ids": list(train_ids), "test_subject_ids": list(test_ids),
                },
            }, _out)
            print(f" saved bundle ({len(_am)} anchor + {len(_pm)} polish) -> {_out}")

        if not test_ids:
            print(" no held-out subjects (train-on-all run): models saved, scoring skipped")
            print(f" fold time: {time.time() - t0:.1f}s")
            continue
        print(" evaluating on held-out test subjects...")
        errs, rows = evaluate_pipeline(ds, template, model, test_ids, polish_model=polish_model,
                                        registration_method=args.registration_method,
                                        tta_views=args.tta_views, multiscale=args.multiscale_patch,
                                        gaussian=args.gaussian_curvature,
                                        crest=args.crest_attract,
                                        members_per_point=args.members_per_point,
                                        geodesic=args.geodesic_patch,
                                        patch_points=args.patch_points,
                                        patch_radius=args.patch_radius,
                                        curve_model=curve_model,
                                        equidistant=args.equidistant,
                                        derive_terminals=args.derive_terminals,
                                        two_pass=args.two_pass)
        all_errors.append(errs)
        if args.save_predictions:
            saved_rows.extend(rows)
        summarize(errs, f"fold {fold_i + 1}")
        from src import pipeline as _pl
        print(f" correction guard triggered {_pl.GUARD_TRIGGERS[0]} times so far "
              f"(corrections > {_pl.MAX_CORRECTION_MM:.0f}mm or non-finite, replaced by the prior)")

        if prior is not None:
            print(" evaluating PAIRED arm with shape prior (same models, same ears)...")
            errs_p, _ = evaluate_pipeline(ds, template, model, test_ids,
                                           polish_model=polish_model,
                                           registration_method=args.registration_method,
                                           tta_views=args.tta_views,
                                           multiscale=args.multiscale_patch,
                                           gaussian=args.gaussian_curvature,
                                           crest=args.crest_attract, shape_prior=prior,
                                           prior_strength=args.prior_strength,
                                           members_per_point=args.members_per_point,
                                        geodesic=args.geodesic_patch,
                                        patch_points=args.patch_points,
                                        patch_radius=args.patch_radius,
                                        curve_model=curve_model)
            prior_errors.append(errs_p)
            summarize(errs_p, f"fold {fold_i + 1} +prior")
        print(f" fold time: {time.time() - t0:.1f}s")

    if not all_errors:
        print("\nno held-out subjects were scored (train-on-all run); see the saved bundle.")
        return
    all_errors = np.concatenate(all_errors, axis=0)
    print("\n" + "=" * 60)
    summarize(all_errors, "OVERALL (all outer folds)")

    if args.save_predictions and saved_rows:
        np.savez_compressed(
            args.save_predictions,
            pred=np.array([r["pred"] for r in saved_rows]),
            truth=np.array([r["truth"] for r in saved_rows]),
            subject_id=np.array([r["subject_id"] for r in saved_rows]),
            side=np.array([r["side"] for r in saved_rows]))
        print(f"\nsaved {len(saved_rows)} ears of predictions -> {args.save_predictions}")

    if prior_errors:
        pe = np.concatenate(prior_errors, axis=0)
        print()
        summarize(pe, "OVERALL +shape prior")
        # PAIRED: both arms share the same trained models and the same ears, so
        # training variance is identical in each and cancels in the difference.
        # This is what makes a ~0.04mm effect measurable against a ~0.11mm
        # training-noise floor.
        d = pe.mean(axis=1) - all_errors.mean(axis=1)
        se = d.std(ddof=1) / np.sqrt(len(d))
        print("\n" + "-" * 60)
        print(f"PAIRED shape-prior effect: {d.mean():+.4f}mm +/- {se:.4f} (SE)")
        print(f"  95% CI [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]   "
              f"ears improved {(d < 0).sum()}/{len(d)}")
        print(f"  -> {'SIGNIFICANT' if abs(d.mean()) > 1.96 * se else 'not significant'}"
              " at 95%   [negative = prior helps]")


if __name__ == "__main__":
    main()
