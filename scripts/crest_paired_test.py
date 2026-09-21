"""Re-test crest attraction as a PAIRED comparison.

The first crest-attraction test compared two SEPARATE training runs, so the
0.012mm difference it produced was swamped by a 0.114mm training-noise floor
and told us nothing. That was avoidable: crest attraction is pure
post-processing on the blend output, so both arms can share one set of trained
models and be evaluated on the same ears. Training variance is then identical
in both arms and cancels exactly in the difference.

Procedure: train once, then for every ear compute blend_correct twice from the
SAME corrected anchors -- once with crest attraction, once without -- and
compare per-ear. Because the arms are paired, a per-ear difference and its
standard error can be computed directly, so the result comes with a proper
uncertainty rather than a single point estimate.

Only the 70 non-anchor points can differ: crest_attract excludes anchors by
construction. The anchor error is therefore reported as a CONTROL -- it must be
identical between arms, and any difference indicates a bug.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correction.anchor_model import predict_correction, train_multi_seed_ensemble
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES, ANCHOR_POSITION, CONTOUR_SPECS, N_LANDMARKS
from src.foundations.dataset import Dataset
from src.interpolation.interpolation import blend_correct
from src.correction.patch_features import crop_submesh, extract_patch
from src.registration.registration import compute_raw_full
from src.foundations.splits import k_fold_subject_split
from src.registration.template import build_template
from src.correction.training_data import generate_inner_cv_examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-subjects", type=int, default=48)
    p.add_argument("--k-references", type=int, default=7)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--capacity", default="large")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                  landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    ids = ds.subject_ids[:args.n_subjects]
    train_ids, test_ids = next(iter(k_fold_subject_split(ids, 2, args.seed)))
    print(f"train {len(train_ids)} / measure {len(test_ids)} subjects "
          f"({args.n_seeds} seeds, capacity={args.capacity})")

    ex = generate_inner_cv_examples(ds, train_ids, k_inner=args.inner_folds,
                                     k_references=args.k_references, seed=args.seed,
                                     verbose=False)
    models = train_multi_seed_ensemble(ex, n_models=args.n_seeds, n_epochs=args.epochs,
                                        capacity=args.capacity, verbose=False)
    template = build_template(ds, train_ids, k_references=args.k_references)
    print("models trained -- both arms will share them\n")

    off_all, on_all, anch_off, anch_on = [], [], [], []
    per_contour = {c: {"off": [], "on": []} for c in CONTOUR_SPECS}

    for exm in iter_landmarks_only(ds, test_ids):
        mesh = load_canonical_mesh(ds, exm.subject_id, exm.side)
        raw = compute_raw_full(template, mesh.vertices, method="tps")
        local = crop_submesh(mesh, template.global_crop_center, template.global_crop_radius)

        corrected = {}
        for a in ANCHOR_INDICES:
            patch = extract_patch(local, raw[a])
            d = np.mean([predict_correction(m, patch, ANCHOR_POSITION[a]) for m in models], axis=0)
            corrected[a] = raw[a] + d

        # identical inputs, only the crest flag differs
        off = blend_correct(raw, corrected, mesh.vertices, crest_mesh=None)
        on = blend_correct(raw, corrected, mesh.vertices, crest_mesh=local)

        e_off = np.linalg.norm(off - exm.landmarks, axis=1)
        e_on = np.linalg.norm(on - exm.landmarks, axis=1)
        off_all.append(e_off.mean()); on_all.append(e_on.mean())
        anch_off.append(e_off[ANCHOR_INDICES].mean())
        anch_on.append(e_on[ANCHOR_INDICES].mean())
        for c, spec in CONTOUR_SPECS.items():
            s, t = spec["range"]
            per_contour[c]["off"].append(e_off[s:t].mean())
            per_contour[c]["on"].append(e_on[s:t].mean())

    off_all = np.array(off_all); on_all = np.array(on_all)
    diff = on_all - off_all
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)

    print(f"ears: {n}\n")
    print("=== PAIRED comparison (same models, same ears) ===")
    print(f"  crest OFF   {off_all.mean():.4f}mm")
    print(f"  crest ON    {on_all.mean():.4f}mm")
    print(f"  difference  {diff.mean():+.4f}mm  +/- {se:.4f} (SE)   "
          f"[negative = crest helps]")
    print(f"  95% CI      [{diff.mean() - 1.96 * se:+.4f}, {diff.mean() + 1.96 * se:+.4f}]")
    print(f"  ears improved: {(diff < 0).sum()}/{n}")
    sig = abs(diff.mean()) > 1.96 * se
    print(f"  -> {'SIGNIFICANT' if sig else 'NOT significant'} at 95%")

    print(f"\n=== control: anchors (crest_attract excludes them) ===")
    ao, an = np.array(anch_off), np.array(anch_on)
    print(f"  OFF {ao.mean():.4f}mm   ON {an.mean():.4f}mm   "
          f"diff {an.mean() - ao.mean():+.6f}mm")
    print("  (must be ~0; anything else means anchors were touched)")

    print("\n=== per contour ===")
    print(f"  {'contour':22s}{'OFF':>10s}{'ON':>10s}{'diff':>11s}")
    print("  " + "-" * 53)
    for c in CONTOUR_SPECS:
        o = np.array(per_contour[c]["off"]); n2 = np.array(per_contour[c]["on"])
        tag = "  <- targeted" if c != "superior_antihelix" else "  <- excluded"
        print(f"  {c:22s}{o.mean():9.4f}m{n2.mean():9.4f}m{n2.mean() - o.mean():+10.4f}m{tag}")

    print("\nFor reference, the unpaired version of this test produced -0.012mm")
    print("against a 0.114mm noise floor, i.e. no usable information.")


if __name__ == "__main__":
    main()
