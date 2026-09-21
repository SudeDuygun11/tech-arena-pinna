"""Can a ranker actually reach the candidate-set ceiling? Estimated WITHOUT
training a ranker, by reusing the existing correction network.

The candidate set (10mm ball, 256 FPS candidates) has an oracle ceiling of
0.542mm -- but that assumes perfect selection. This estimates the realisable
number three ways:

  TEST 1 proxy ranker
    The correction net answers "from patch at p, what moves it to truth?".
    If p IS truth the correction should be ~0, so |predicted correction| is a
    free candidate score. Pick argmin. This is PESSIMISTIC -- the net was
    trained to regress, not to discriminate candidates -- so treat it as a
    lower bound on what a purpose-trained ranker would achieve.

  TEST 2 consensus voting
    Every candidate votes for where truth is (p + delta(p)). Aggregate the
    votes (median / trimmed mean). Errors from bad candidates partially
    cancel while good ones agree.

  TEST 3 separability
    Spearman correlation between |predicted correction| and the candidate's
    true distance to truth. This measures whether the SIGNAL a ranker needs
    exists at all, independent of how well this particular net exploits it.

Reference points printed alongside: the oracle ceiling, a random pick (upper
bound), and the current pipeline's 1.5931mm.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import spearmanr

from src.correction.anchor_model import predict_correction, train_model
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, ANCHOR_POSITION, contour_of
from src.foundations.dataset import Dataset
from src.correction.patch_features import crop_submesh, extract_patch
from src.registration.registration import _farthest_point_sample, compute_raw_full
from src.foundations.splits import k_fold_subject_split
from src.registration.template import build_template
from src.correction.training_data import generate_inner_cv_examples


def summarise(name, errs, extra=""):
    e = np.array(errs)
    print(f"  {name:34s} mean {e.mean():7.3f}mm   median {np.median(e):7.3f}mm   "
          f"p95 {np.percentile(e, 95):7.3f}mm {extra}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=24)
    p.add_argument("--k-references", type=int, default=7)
    p.add_argument("--inner-folds", type=int, default=2)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--capacity", default="large")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-seed", type=int, default=0)
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    ids = ds.subject_ids[:args.n_subjects]
    train_ids, test_ids = next(iter(k_fold_subject_split(ids, 2, args.seed)))
    print(f"train {len(train_ids)} / measure {len(test_ids)}   "
          f"radius={args.radius}mm budget={args.budget}")

    ex = generate_inner_cv_examples(ds, train_ids, k_inner=args.inner_folds,
                                     k_references=args.k_references, seed=args.seed,
                                     verbose=False)
    model = train_model(ex, n_epochs=args.epochs, capacity=args.capacity,
                         verbose=False, torch_seed=args.torch_seed)
    template = build_template(ds, train_ids, k_references=args.k_references)
    print("correction model trained\n")

    oracle, proxy, vote_med, vote_trim, rnd, seed_err = [], [], [], [], [], []
    rho_per_anchor = []
    all_score, all_truth_d = [], []
    per_contour = {}
    rng = np.random.default_rng(args.seed)

    for exm in iter_landmarks_only(ds, test_ids):
        mesh = load_canonical_mesh(ds, exm.subject_id, exm.side)
        raw = compute_raw_full(template, mesh.vertices, method="tps")
        local = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)
        V = np.asarray(local.vertices)

        for a in ANCHOR_INDICES:
            s, t = raw[a], exm.landmarks[a]
            inside = np.linalg.norm(V - s[None, :], axis=1) <= args.radius
            if inside.sum() < 8:
                continue
            pts = V[inside]
            if len(pts) > args.budget:
                pts = pts[_farthest_point_sample(pts, args.budget, seed=args.seed)]

            d_true = np.linalg.norm(pts - t[None, :], axis=1)
            deltas = np.array([predict_correction(model, extract_patch(local, c),
                                                   ANCHOR_POSITION[a]) for c in pts])
            score = np.linalg.norm(deltas, axis=1)      # small = "already correct"
            votes = pts + deltas

            seed_err.append(float(np.linalg.norm(s - t)))
            oracle.append(float(d_true.min()))
            proxy.append(float(d_true[int(np.argmin(score))]))
            vote_med.append(float(np.linalg.norm(np.median(votes, axis=0) - t)))
            keep = np.argsort(score)[:max(3, args.budget // 10)]
            vote_trim.append(float(np.linalg.norm(votes[keep].mean(axis=0) - t)))
            rnd.append(float(d_true[rng.integers(len(pts))]))

            if np.std(score) > 1e-9:
                rho_per_anchor.append(spearmanr(score, d_true).statistic)
            all_score.extend(score.tolist()); all_truth_d.extend(d_true.tolist())

            c = contour_of(a)
            per_contour.setdefault(c, {"oracle": [], "proxy": [], "vote_trim": []})
            per_contour[c]["oracle"].append(float(d_true.min()))
            per_contour[c]["proxy"].append(float(d_true[int(np.argmin(score))]))
            per_contour[c]["vote_trim"].append(vote_trim[-1])

    print(f"anchors measured: {len(oracle)}\n")
    print("=== realisable accuracy estimates ===")
    summarise("ORACLE (perfect ranking)", oracle, "<- ceiling")
    summarise("TEST 1  proxy ranker (argmin)", proxy, "<- pessimistic lower bound")
    summarise("TEST 2a consensus (median vote)", vote_med)
    summarise("TEST 2b consensus (top-10% trimmed)", vote_trim)
    summarise("random candidate pick", rnd, "<- upper bound")
    summarise("registration seed (no ranking)", seed_err)
    print(f"  {'current full pipeline':34s} mean   1.593mm   (200-subject 3-fold CV reference)")

    rho = np.array([r for r in rho_per_anchor if np.isfinite(r)])
    gr = spearmanr(all_score, all_truth_d).statistic
    print("\n=== TEST 3  separability ===")
    print(f"  per-anchor Spearman(|correction|, true distance): "
          f"mean {rho.mean():+.3f}   median {np.median(rho):+.3f}")
    print(f"  fraction of anchors with positive correlation: "
          f"{(rho > 0).mean() * 100:.1f}%")
    print(f"  pooled Spearman across all candidates: {gr:+.3f}")
    print("  (positive => candidates nearer truth get lower scores, i.e. the")
    print("   signal a ranker needs is present; ~0 => candidates are")
    print("   indistinguishable and ranking cannot work)")

    print("\n=== per contour (mean mm) ===")
    hdr = f"{'contour':22s}{'oracle':>10s}{'proxy':>10s}{'vote_trim':>12s}"
    print(hdr); print("-" * len(hdr))
    for c in sorted(per_contour):
        d = per_contour[c]
        print(f"{c:22s}{np.mean(d['oracle']):9.3f}m{np.mean(d['proxy']):9.3f}m"
              f"{np.mean(d['vote_trim']):11.3f}m")


if __name__ == "__main__":
    main()
