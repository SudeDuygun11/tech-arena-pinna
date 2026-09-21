"""Decision 3 test: how strongly may the shape prior override local evidence?

Research (Bayesian AAM/ASM formulations; see also 'A weighting strategy for
Active Shape Models') says this is NOT a free knob: under a MAP formulation the
regularisation weight is the INVERSE COVARIANCE of the prior, and each landmark
enters weighted by its own measurement covariance. So the principled answer is
inverse-variance weighting -- provided we can estimate per-anchor measurement
variance, which Decision 2 gave us via E-CPV (Spearman +0.310).

Caveat that this script has to handle: E-CPV was shown to CORRELATE with error,
not to be CALIBRATED to it. Inverse-variance weighting needs an actual variance,
so E-CPV is first calibrated to error magnitude on a held-out split.

Model fitted: observe the 15 anchors (noisy ensemble predictions), fit the
85-point PCA shape model to those observations, read back the constrained
anchor positions. Decision 1 established the 85-point prior is the sharper and
more constraining one (0.377mm at 45/255 modes).

Compared:
  local only            -- ensemble prediction, no prior          (baseline)
  prior sweep           -- uniform weights, prior strength varied (empirical)
  MAP + calibrated ECPV -- per-anchor inverse-variance weighting  (principled)
  prior only            -- ignore local confidence entirely
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correction.anchor_model import predict_correction, train_multi_seed_ensemble
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, ANCHOR_POSITION
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid, kabsch
from src.correction.patch_features import crop_submesh, extract_patch
from src.registration.registration import compute_raw_full
from src.foundations.splits import k_fold_subject_split
from src.registration.template import build_template
from src.foundations.contours import N_LANDMARKS
from src.interpolation.interpolation import blend_correct
from src.correction.training_data import generate_inner_cv_examples, generate_polish_examples

N_MODES = 45
STRENGTHS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0, 1e6]


def procrustes_align(shapes, n_iter=5):
    aligned = shapes.copy()
    mean = aligned[0]
    for _ in range(n_iter):
        for i in range(len(aligned)):
            R, t = kabsch(aligned[i], mean)
            aligned[i] = apply_rigid(aligned[i], R, t)
        mean = aligned.mean(axis=0)
    return aligned, mean


def fit_constrained(obs, obs_rows, mu, P, evals, strength, weights=None):
    """Weighted MAP fit of shape coefficients to partial observations.

    b = (P_o^T W P_o + strength * Lambda^-1)^-1 P_o^T W (obs - mu_o)
    strength -> 0   : prior ignored (pure least squares on observations)
    strength -> inf : coefficients forced to 0, i.e. the mean shape
    """
    Po = P[:, obs_rows]                      # (k, n_obs_dims)
    mo = mu[obs_rows]
    r = obs.reshape(-1) - mo
    n_obs = len(obs_rows)
    W = np.ones(n_obs) if weights is None else weights
    PW = Po * W[None, :]
    A = PW @ Po.T + strength * np.diag(1.0 / np.maximum(evals, 1e-9))
    b = np.linalg.solve(A, PW @ r)
    return (mu + P.T @ b).reshape(-1, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=24)
    p.add_argument("--k-references", type=int, default=7)
    p.add_argument("--inner-folds", type=int, default=2)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--capacity", default="large")
    p.add_argument("--with-polish", action="store_true",
                    help="feed the FULL pipeline output (reg -> correction -> blend -> "
                         "polish) as the local observation, instead of bare Stage-1b. "
                         "This is what makes the local model strong enough to be "
                         "representative of the real 1.593mm pipeline.")
    p.add_argument("--prior-subjects", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    ids = ds.subject_ids[:args.n_subjects]
    train_ids, test_ids = next(iter(k_fold_subject_split(ids, 2, args.seed)))
    test_set = set(test_ids)

    # --- shape prior, fitted only on subjects NOT in the test split
    prior_shapes = [ex.landmarks for ex in
                     iter_landmarks_only(ds, [s for s in ds.subject_ids[:args.prior_subjects]
                                               if s not in test_set])]
    prior_shapes = np.array(prior_shapes)
    aligned, mean_shape = procrustes_align(prior_shapes.copy())
    X = aligned.reshape(len(aligned), -1)
    mu = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:N_MODES]
    evals = (S[:N_MODES] ** 2) / (len(X) - 1)
    print(f"prior: {len(prior_shapes)} shapes, {N_MODES} modes, "
          f"{100 * np.cumsum(S**2/ (S**2).sum())[N_MODES-1]:.1f}% variance")

    obs_rows = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in ANCHOR_INDICES])

    # --- local model
    ex = generate_inner_cv_examples(ds, train_ids, k_inner=args.inner_folds,
                                     k_references=args.k_references, seed=args.seed,
                                     verbose=False)
    models = train_multi_seed_ensemble(ex, n_models=args.n_seeds, n_epochs=args.epochs,
                                        capacity=args.capacity, verbose=False)
    template = build_template(ds, train_ids, k_references=args.k_references)
    polish = None
    if args.with_polish:
        pex = generate_polish_examples(ds, template, models, train_ids, seed=args.seed)
        polish = train_multi_seed_ensemble(pex, n_models=args.n_seeds,
                                            n_epochs=args.epochs, n_classes=N_LANDMARKS,
                                            capacity=args.capacity, verbose=False)
    print(f"local ensemble trained ({len(models)} models, "
          f"polish={'yes' if polish else 'no'})\n")

    recs = []
    for exm in iter_landmarks_only(ds, test_ids):
        mesh = load_canonical_mesh(ds, exm.subject_id, exm.side)
        raw = compute_raw_full(template, mesh.vertices, method="tps")
        local = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)
        pred, ecpv, corrected = [], [], {}
        for a in ANCHOR_INDICES:
            patch = extract_patch(local, raw[a])
            d = np.array([predict_correction(m, patch, ANCHOR_POSITION[a]) for m in models])
            pr = raw[a] + d
            pred.append(pr.mean(axis=0))
            corrected[a] = pr.mean(axis=0)
            ecpv.append(float(np.linalg.norm(pr - pr.mean(axis=0), axis=1).mean()))
        if polish is not None:
            interp = blend_correct(raw, corrected, mesh.vertices)
            pred = []
            for a in ANCHOR_INDICES:
                patch = extract_patch(local, interp[a])
                d = np.array([predict_correction(m, patch, a) for m in polish])
                pred.append(interp[a] + d.mean(axis=0))
        recs.append(dict(pred=np.array(pred), ecpv=np.array(ecpv),
                          truth=exm.landmarks[ANCHOR_INDICES], full=exm.landmarks))

    base = np.array([np.linalg.norm(r["pred"] - r["truth"], axis=1).mean() for r in recs])
    print(f"anchors: {len(recs) * len(ANCHOR_INDICES)}")
    print(f"LOCAL ONLY (baseline)          {base.mean():.3f}mm\n")

    # calibrate ECPV -> error magnitude on the first half of test ears
    half = len(recs) // 2
    cx = np.concatenate([r["ecpv"] for r in recs[:half]])
    cy = np.concatenate([np.linalg.norm(r["pred"] - r["truth"], axis=1) for r in recs[:half]])
    A = np.polyfit(cx, cy, 1)
    print(f"E-CPV calibration on held-out half: err ~ {A[0]:.3f} * ecpv + {A[1]:.3f}\n")
    eval_recs = recs[half:]
    base_e = np.array([np.linalg.norm(r["pred"] - r["truth"], axis=1).mean() for r in eval_recs])

    print(f"{'prior strength':>16s}{'uniform W':>14s}{'MAP + calib ECPV':>20s}")
    print("-" * 50)
    for st in STRENGTHS:
        u, m = [], []
        for r in eval_recs:
            aligned_pred, _ = None, None
            R, t = kabsch(r["pred"], mean_shape[ANCHOR_INDICES])
            pa = apply_rigid(r["pred"], R, t)
            ta = apply_rigid(r["truth"], R, t)
            fu = fit_constrained(pa, obs_rows, mu, P, evals, st)
            u.append(np.linalg.norm(fu[ANCHOR_INDICES] - ta, axis=1).mean())
            sig = np.maximum(A[0] * r["ecpv"] + A[1], 1e-3)
            w = np.repeat(1.0 / sig ** 2, 3)
            w = w / w.mean()
            fm = fit_constrained(pa, obs_rows, mu, P, evals, st, weights=w)
            m.append(np.linalg.norm(fm[ANCHOR_INDICES] - ta, axis=1).mean())
        lbl = "inf (mean shape)" if st > 1e5 else f"{st:g}"
        print(f"{lbl:>16s}{np.mean(u):13.3f}m{np.mean(m):19.3f}m")

    print(f"\n{'local only (same ears)':>16s}{base_e.mean():13.3f}m")
    print("\nIf no prior strength beats 'local only', the prior cannot help at this")
    print("local accuracy, and Decision 3 is: do not use a shape prior.")


if __name__ == "__main__":
    main()
