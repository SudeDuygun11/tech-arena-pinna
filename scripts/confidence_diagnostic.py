"""Decision 2 test: can we estimate per-anchor CONFIDENCE well enough to be useful?

The global (shape-prior) step needs to know which anchors to trust. Without a
usable confidence signal it must weight all 15 equally, which throws away the
distinction between a well-localised anchor and a badly-guessed one.

Evaluated by the criterion stated in Uncertainty Estimation for Heatmap-based
Landmark Localization (Schobs & Lu, IEEE TMI 2022, arXiv:2203.02351):
"A good uncertainty measure will have a strong correlation with localization
error." So: Spearman(confidence, actual error).

Two measures, both free (no retraining, no architecture change):

  E-CPV  ensemble coordinate prediction variance -- spread of the ensemble
         members' predictions. The paper's STRONG baseline. Their other two
         measures (S-MHA, E-MHA) are heatmap-derived and do not apply to us,
         since we regress coordinates directly.

  CORR   magnitude of the predicted correction. If the patch is already
         centred on the landmark the correction should be ~0. Previously
         measured at Spearman +0.301 (93.9% of anchors positive) as a
         candidate ranker, so there is a baseline to beat.

Also reports quantile binning (the paper's practical framing): if predictions
are bucketed by confidence, does error actually rise across the buckets? A
measure can correlate weakly yet still separate the best bucket usefully.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import spearmanr

from src.correction.anchor_model import predict_correction, train_multi_seed_ensemble
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, ANCHOR_POSITION, contour_of
from src.foundations.dataset import Dataset
from src.correction.patch_features import crop_submesh, extract_patch
from src.registration.registration import compute_raw_full
from src.foundations.splits import k_fold_subject_split
from src.registration.template import build_template
from src.correction.training_data import generate_inner_cv_examples

N_BINS = 5


def quantile_report(name, conf, err):
    order = np.argsort(conf)
    bins = np.array_split(order, N_BINS)
    print(f"\n  {name} -- error by confidence quintile (B1 = most confident):")
    parts = []
    for i, b in enumerate(bins):
        parts.append(f"B{i + 1} {err[b].mean():.3f}mm")
    print("    " + "   ".join(parts))
    lo, hi = err[bins[0]].mean(), err[bins[-1]].mean()
    print(f"    spread B5-B1 = {hi - lo:+.3f}mm  (positive = measure works)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=24)
    p.add_argument("--k-references", type=int, default=7)
    p.add_argument("--inner-folds", type=int, default=2)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--capacity", default="large")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    ids = ds.subject_ids[:args.n_subjects]
    train_ids, test_ids = next(iter(k_fold_subject_split(ids, 2, args.seed)))
    print(f"train {len(train_ids)} / measure {len(test_ids)}   "
          f"{args.n_seeds}-model ensemble, capacity={args.capacity}")

    ex = generate_inner_cv_examples(ds, train_ids, k_inner=args.inner_folds,
                                     k_references=args.k_references, seed=args.seed,
                                     verbose=False)
    models = train_multi_seed_ensemble(ex, n_models=args.n_seeds, n_epochs=args.epochs,
                                        capacity=args.capacity, verbose=False)
    template = build_template(ds, train_ids, k_references=args.k_references)
    print(f"ensemble trained ({len(models)} models)\n")

    ecpv, corrmag, err, cname = [], [], [], []

    for exm in iter_landmarks_only(ds, test_ids):
        mesh = load_canonical_mesh(ds, exm.subject_id, exm.side)
        raw = compute_raw_full(template, mesh.vertices, method="tps")
        local = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)

        for a in ANCHOR_INDICES:
            patch = extract_patch(local, raw[a])
            deltas = np.array([predict_correction(m, patch, ANCHOR_POSITION[a])
                               for m in models])
            preds = raw[a] + deltas
            mean_pred = preds.mean(axis=0)

            ecpv.append(float(np.linalg.norm(preds - mean_pred, axis=1).mean()))
            corrmag.append(float(np.linalg.norm(deltas.mean(axis=0))))
            err.append(float(np.linalg.norm(mean_pred - exm.landmarks[a])))
            cname.append(contour_of(a))

    ecpv = np.array(ecpv); corrmag = np.array(corrmag); err = np.array(err)
    cname = np.array(cname)

    print(f"anchors measured: {len(err)}")
    print(f"ensemble-mean prediction error: mean {err.mean():.3f}mm  "
          f"median {np.median(err):.3f}mm  p95 {np.percentile(err, 95):.3f}mm")

    print("\n=== Spearman(confidence measure, actual error) ===")
    print("    higher = measure predicts error better; ~0 = useless")
    r_e = spearmanr(ecpv, err).statistic
    r_c = spearmanr(corrmag, err).statistic
    print(f"  E-CPV  (ensemble variance)      {r_e:+.3f}")
    print(f"  CORR   (|predicted correction|) {r_c:+.3f}   [prior measurement: +0.301]")

    print("\n=== per contour ===")
    print(f"  {'contour':22s}{'E-CPV':>10s}{'CORR':>10s}{'mean err':>12s}")
    print("  " + "-" * 52)
    for c in sorted(set(cname)):
        m = cname == c
        print(f"  {c:22s}{spearmanr(ecpv[m], err[m]).statistic:+10.3f}"
              f"{spearmanr(corrmag[m], err[m]).statistic:+10.3f}{err[m].mean():11.3f}m")

    print("\n=== quantile binning (practical usefulness) ===")
    quantile_report("E-CPV", ecpv, err)
    quantile_report("CORR ", corrmag, err)

    print("\nVerdict guide: a measure needs a clearly positive correlation AND a")
    print("real B5-B1 spread to be worth feeding to the global step. If both are")
    print("weak, Decision 2 should be POSITION ONLY.")


if __name__ == "__main__":
    main()
