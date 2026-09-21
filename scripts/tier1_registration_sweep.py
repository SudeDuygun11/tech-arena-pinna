"""Tier 1: settle every registration/template constant WITHOUT training.

WHY THIS IS CHEAP
-----------------
Stage 1a and the template are scored with predict_canonical(template, None, mesh)
-- registration only, no network. That is roughly ten times cheaper than a full
cross-validation run, and it covers seven of the eighteen constants that the
provenance audit lists as having no evidence behind them:

    TPS_SMOOTHING, TRIM_FRACTION, RIGID_ICP_ITERS, NONRIGID_ICP_ITERS,
    N_CONTROL_SURFACE_POINTS, k_references, reference selection, CROP_MARGIN

They are also UPSTREAM: every learned-stage result sits on top of whatever
registration produces, so settling these first makes the expensive Tier 2 sweeps
cleaner rather than merely earlier.

WHAT IS MEASURED
----------------
Mean 85-point error of the registration prior alone, on held-out subjects. The
learned correction is deliberately absent -- this isolates the prior's quality.
A better prior is not guaranteed to produce a better final number (the network
exists to fix the prior's mistakes), so a win here is a candidate for a Tier 2
confirmation, not a finished result.

Constants are module-level, so they are monkeypatched per variant. The template
is rebuilt only when a variant actually changes it, since that is the expensive
part.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES
from src.foundations.dataset import Dataset
from src.foundations.geometry import apply_rigid, kabsch
from src.pipeline import predict_canonical
from src.registration import registration as R
from src.registration import template as T


def set_default(fn, name, value):
    """Override a function's DEFAULT ARGUMENT.

    Necessary because registration binds its constants as default arguments --
    `def _rigid_icp(..., n_iters=RIGID_ICP_ITERS)` -- which Python evaluates
    once at import. Reassigning the module global afterwards has NO EFFECT, so
    a sweep that patches R.TPS_SMOOTHING silently measures nothing and reports
    a flat line. That is exactly what happened on the first run of this script.
    """
    import inspect
    names = [q.name for q in inspect.signature(fn).parameters.values()
             if q.default is not inspect.Parameter.empty]
    d = list(fn.__defaults__)
    d[names.index(name)] = value
    fn.__defaults__ = tuple(d)


def score(template, ds, test_ids, cache):
    errs = []
    for sid, side, L, mesh in cache:
        pred = predict_canonical(template, None, mesh)
        errs.append(np.linalg.norm(pred - L, axis=1).mean())
    return float(np.mean(errs))


def select_references(ds, train_ids, k, mode):
    """Return the k reference indices under a given selection rule."""
    ex = list(iter_landmarks_only(ds, train_ids))
    A = [np.asarray(e.landmarks, float)[ANCHOR_INDICES] for e in ex]
    mean = T._gpa_mean_anchor_shape(A)
    aligned = [apply_rigid(a, *kabsch(a, mean)) for a in A]
    res = np.array([np.mean(np.linalg.norm(al - mean, axis=1)) for al in aligned])
    if mode == "medoid":
        return list(np.argsort(res)[:k])
    if mode == "random":
        return list(np.random.default_rng(0).choice(len(A), k, replace=False))
    # farthest-point: start from the medoid, then maximise minimum distance
    sel = [int(np.argmin(res))]
    while len(sel) < k:
        best, bd = None, -1.0
        for i in range(len(A)):
            if i in sel:
                continue
            dmin = min(np.mean(np.linalg.norm(aligned[i] - aligned[j], axis=1)) for j in sel)
            if dmin > bd:
                bd, best = dmin, i
        sel.append(best)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=15)
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    train_ids = ds.subject_ids[:args.n_train]
    test_ids = ds.subject_ids[args.n_train:args.n_train + args.n_test]

    print(f"template pool {len(train_ids)} subjects, test {len(test_ids)} subjects")
    print("loading test meshes once...", flush=True)
    cache = []
    for e in iter_landmarks_only(ds, test_ids):
        cache.append((e.subject_id, e.side, np.asarray(e.landmarks, float),
                      load_canonical_mesh(ds, e.subject_id, e.side)))
    print(f"  {len(cache)} ears held in memory\n", flush=True)

    base_tpl = T.build_template(ds, train_ids, k_references=7)
    t0 = time.time()
    base = score(base_tpl, ds, test_ids, cache)
    print(f"BASELINE (current settings): {base:.4f} mm   [{time.time()-t0:.0f}s per config]\n",
          flush=True)

    def run(label, value, restore):
        s = score(base_tpl, ds, test_ids, cache)
        print(f"  {label:<34s} {value:>8} -> {s:7.4f} mm  ({s-base:+.4f})", flush=True)
        restore()
        return s

    # ---- registration constants: template unchanged, so no rebuild needed
    print("TPS_SMOOTHING  (currently 4.0, never tuned; GCV is the principled choice)")
    for v in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        set_default(R._nonrigid_refine, "smoothing", v)
        run("TPS_SMOOTHING", v, lambda: None)
    set_default(R._nonrigid_refine, "smoothing", R.TPS_SMOOTHING)

    print("\nTRIM_FRACTION  (currently 0.15; should track non-overlap, ours is high-overlap)")
    old = R.TRIM_FRACTION
    for v in (0.0, 0.05, 0.15, 0.30):
        R.TRIM_FRACTION = v
        run("TRIM_FRACTION", v, lambda: None)
    R.TRIM_FRACTION = old

    print("\nRIGID_ICP_ITERS  (currently 8; convergence never checked)")
    for v in (2, 4, 8, 16):
        set_default(R._rigid_icp, "n_iters", v)
        run("RIGID_ICP_ITERS", v, lambda: None)
    set_default(R._rigid_icp, "n_iters", R.RIGID_ICP_ITERS)

    print("\nNONRIGID_ICP_ITERS  (currently 6)")
    for v in (2, 4, 6, 10):
        set_default(R._nonrigid_refine, "n_iters", v)
        run("NONRIGID_ICP_ITERS", v, lambda: None)
    set_default(R._nonrigid_refine, "n_iters", R.NONRIGID_ICP_ITERS)

    print("\nN_CONTROL_SURFACE_POINTS  (currently 450)")
    old = R.N_CONTROL_SURFACE_POINTS
    for v in (150, 450, 900):
        R.N_CONTROL_SURFACE_POINTS = v
        tpl = T.build_template(ds, train_ids, k_references=7)
        s = score(tpl, ds, test_ids, cache)
        print(f"  {'N_CONTROL_SURFACE_POINTS':<34s} {v:>8} -> {s:7.4f} mm  ({s-base:+.4f})",
              flush=True)
    R.N_CONTROL_SURFACE_POINTS = old

    # ---- template constants: rebuild required
    print("\nCROP_MARGIN  (currently 18.0; zero mentions in any results document)")
    for v in (8.0, 18.0, 30.0):
        tpl = T.build_template(ds, train_ids, k_references=7, crop_margin=v)
        s = score(tpl, ds, test_ids, cache)
        print(f"  {'CROP_MARGIN':<34s} {v:>8} -> {s:7.4f} mm  ({s-base:+.4f})", flush=True)

    print("\nk_references  (currently 7; module default is 5, never isolated)")
    for v in (3, 5, 7, 11, 15):
        tpl = T.build_template(ds, train_ids, k_references=v)
        s = score(tpl, ds, test_ids, cache)
        print(f"  {'k_references':<34s} {v:>8} -> {s:7.4f} mm  ({s-base:+.4f})", flush=True)

    print("\nreference selection  (medoid-7 spread 1.878mm vs 5.744 for farthest-point)")
    ex = list(iter_landmarks_only(ds, train_ids))
    for mode in ("medoid", "random", "farthest"):
        idx = select_references(ds, train_ids, 7, mode)
        refs = []
        for i in idx:
            e = ex[i]
            V = np.asarray(load_canonical_mesh(ds, e.subject_id, e.side).vertices)
            c = e.landmarks.mean(axis=0)
            r = np.max(np.linalg.norm(e.landmarks - c, axis=1)) + T.CROP_MARGIN
            refs.append(R.build_reference(f"{e.subject_id}_{e.side}", e.landmarks, V, c, r))
        tpl = R.Template(references=refs,
                         global_crop_center=base_tpl.global_crop_center,
                         global_crop_radius=base_tpl.global_crop_radius)
        s = score(tpl, ds, test_ids, cache)
        print(f"  {'reference selection':<34s} {mode:>8} -> {s:7.4f} mm  ({s-base:+.4f})",
              flush=True)

    print(f"\nbaseline was {base:.4f} mm -- anything better than that by more than run-to-run")
    print("noise is a Tier 2 candidate, not a finished result.")


if __name__ == "__main__":
    main()
