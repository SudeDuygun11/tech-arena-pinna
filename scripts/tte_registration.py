"""Test-time ensembling over REGISTRATION diversity, with one trained model.

Finding that motivates this: averaging two separately trained pipelines (one-pass
1.3736, two-pass 1.3653) gives 1.3314 (t = -6.86), and the relational T2
correction stacks on top almost additively (1.2905, t = -9.25). Diversity of
predictions is a real lever -- while every added INPUT feature overfit.

Question: can that diversity be produced at inference time, from ONE trained
model, by varying only the registration prior the model starts from? If yes,
it costs no training and scales with the number of variants.

Uses the saved fold-3 bundle (assets/bundle_twopass.pt, outer_fold=2) on that
fold's 132 held-out ears -- reconstructed from the same seed-0 subject split, so
no scored ear was seen in training.

Variants (same anchor + polish ensembles throughout):
  P2  two registration passes (the trained configuration)
  P1  one registration pass
  JLF one pass, prior = per-landmark locally weighted fusion of the 11 reference
      registrations (joint-label-fusion style; improved the registration prior
      3.342 -> 3.217mm in multiatlas_features.py)
  B1..B3  one pass, prior from a random 7-of-11 subset of references (bagging)
Then every average of these.
"""
from __future__ import annotations

import copy
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import src.pipeline as pipeline
from src.estimator import _build
from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import CONTOUR_SPECS, N_ANCHORS, N_LANDMARKS
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid
from src.foundations.splits import k_fold_subject_split
from src.interpolation.interpolation import resample_uniform
from src.registration import registration as R

DATA = ROOT / "2026 Munich Tech Arena - Datas"
ORIG_RAW = pipeline.compute_raw_full


def jlf_prior(template, V, method="tps", pinned_anchors=None):
    preds, locs = [], []
    c, rad = template.global_crop_center, template.global_crop_radius
    dist = np.linalg.norm(V - c[None, :], axis=1)
    cropped = V[dist <= rad]
    if len(cropped) < 20:
        cropped = V[dist <= rad * 2]
    tree = cKDTree(cropped)
    for ref in template.references:
        Rm, t, rs = R._rigid_icp(ref.control_points, cropped)
        spline, _ = R._nonrigid_refine(rs, cropped)
        lm = spline(apply_rigid(ref.landmarks, Rm, t)); cp = spline(rs)
        d, _ = tree.query(cp)
        loc = np.array([d[np.linalg.norm(cp - p, axis=1) <= 5.0].mean()
                        if (np.linalg.norm(cp - p, axis=1) <= 5.0).any() else d.mean() for p in lm])
        preds.append(lm); locs.append(loc)
    preds, locs = np.array(preds), np.array(locs)
    w = np.exp(-locs / np.median(locs)); w /= w.sum(0, keepdims=True)
    return (preds * w[..., None]).sum(0)


def respace(P):
    out = P.copy()
    for spec in CONTOUR_SPECS.values():
        a = spec["anchors"]
        for u, v in zip(a[:-1], a[1:]):
            out[u:v + 1] = resample_uniform(out[u:v + 1])
    return out


def main():
    b = torch.load(ROOT / "assets" / "bundle_twopass.pt", map_location="cpu", weights_only=False)
    cfg = b["config"]; template = b["template"]
    am = [_build(s, N_ANCHORS) for s in b["anchor_models"]]
    pm = [_build(s, N_LANDMARKS) for s in b["polish_models"]]
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    _, test_ids = list(k_fold_subject_split(ds.subject_ids[:cfg.get("n_subjects", 200)], 3, 0))[cfg["outer_fold"]]
    rng = np.random.default_rng(0)
    bags = []
    for _ in range(3):
        t = copy.copy(template)
        t.references = [template.references[j] for j in sorted(rng.choice(len(template.references), 7, replace=False))]
        bags.append(t)
    print(f"fold {cfg['outer_fold']+1}: {len(test_ids)} held-out subjects; {len(template.references)} references", flush=True)

    kw = dict(polish_model=pm, registration_method="tps", gaussian=cfg.get("gaussian_curvature", False),
              geodesic=cfg.get("geodesic_patch", False), patch_points=cfg.get("patch_points"),
              patch_radius=cfg.get("patch_radius"))
    names = ["P2", "P1", "JLF", "B1", "B2", "B3"]
    preds = {nm: [] for nm in names}; truth = []
    t0 = time.time()
    ears = list(iter_landmarks_only(ds, test_ids))
    for ei, e in enumerate(ears):
        mesh = load_canonical_mesh(ds, e.subject_id, e.side)
        truth.append(np.asarray(e.landmarks, float))
        preds["P2"].append(pipeline.predict_canonical(template, am, mesh, n_passes=2, **kw))
        preds["P1"].append(pipeline.predict_canonical(template, am, mesh, n_passes=1, **kw))
        pipeline.compute_raw_full = jlf_prior
        try:
            preds["JLF"].append(pipeline.predict_canonical(template, am, mesh, n_passes=1, **kw))
        finally:
            pipeline.compute_raw_full = ORIG_RAW
        for j, t in enumerate(bags):
            preds[f"B{j+1}"].append(pipeline.predict_canonical(t, am, mesh, n_passes=1, **kw))
        del mesh
        if (ei + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {ei+1}/{len(ears)} ears  {el/60:.1f} min  (~{el/(ei+1)*(len(ears)-ei-1)/60:.0f} min left)", flush=True)
    truth = np.array(truth); preds = {k: np.array(v) for k, v in preds.items()}
    np.savez_compressed(ROOT / "results" / "tte_registration.npz", truth=truth, **preds)

    err = lambda P: np.linalg.norm(P - truth, axis=2).mean(1)
    ref = err(preds["P2"]); n = len(truth)
    print(f"\n{len(truth)} ears. single variants:")
    for nm in names:
        d = err(preds[nm]) - ref
        print(f"  {nm:4s} {err(preds[nm]).mean():.4f}   vs P2 {d.mean():+.4f} (t {d.mean()/(d.std(ddof=1)/np.sqrt(n)+1e-12):+.2f})")
    print("\naverages (respaced):")
    rows = []
    for r in range(2, len(names) + 1):
        for combo in combinations(names, r):
            avg = np.array([respace(p) for p in np.mean([preds[c] for c in combo], axis=0)])
            e_ = err(avg); d = e_ - ref
            rows.append((e_.mean(), combo, d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(n))))
    for m, combo, dm, t in sorted(rows)[:10]:
        print(f"  {'+'.join(combo):22s} {m:.4f}   vs P2 {dm:+.4f} (t {t:+.2f})")
    allavg = [r for r in rows if len(r[1]) == len(names)][0]
    print(f"  (all six: {allavg[0]:.4f}, vs P2 {allavg[2]:+.4f}, t {allavg[3]:+.2f})")


if __name__ == "__main__":
    main()
