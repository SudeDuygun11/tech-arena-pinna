"""Medical-literature feature: the helix-to-head distance profile.

Otoplasty norms (StatPearls, "Otoplasty"; plastic-surgery literature): the
helical rim sits 10-12mm from the mastoid scalp in the upper third, 16-18mm in
the middle third and 20-22mm in the lower third; the auriculocephalic angle is
20-30 degrees. That is a quantity that CHANGES STEADILY ALONG the outer helix --
exactly the along-contour signal that every local surface feature in rounds
1-4 failed to provide. If the distance to the head encodes where along the helix
a point sits, it could locate the outer-helix anchors (0, 6, 22, 24), including
anchor 22, which no method has improved so far.

MEASUREMENT (test-time information only)
  For every PREDICTED contour point: march a ray from the point toward the head
  along the inward ear-plane normal (from multiview.ear_frame on the predicted
  landmarks). The first surface sample that faces OUTWARD (vertex normal . n_out
  > 0.3) at least 3mm in is the scalp -- this skips the back face of the
  auricle itself, whose normals face the head. Distance recorded in mm; 40mm if
  nothing is hit. Profiles for the outer helix (25 pts), inner helix (20) and
  superior antihelix (10).

TEST
  Same fully nested protocol as feature_probe_r4.py, V1 relational ridge vs V1 +
  head-distance profile (+ its auriculocephalic-angle proxy).
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from feature_probe_r2 import fit_predict
from feature_probe_r3 import AI, STEPS, load_sorted, rebuild
from src.foundations.canonical import load_canonical_mesh
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split
from src.multiview.multiview import ear_frame

DATA = ROOT / "2026 Munich Tech Arena - Datas"


def head_profiles(pred, subj, side, keys):
    cache = ROOT / "results" / "head_distance_profiles.npz"
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        if list(c["keys"]) == list(keys):
            return c["prof"], c["angle"]
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    n = len(pred)
    idx = np.arange(0, 85)                                  # all contour points
    prof = np.full((n, 85), 40.0); angle = np.zeros(n)
    ts = np.arange(3.0, 40.0, 0.5)
    for i in range(n):
        m = load_canonical_mesh(ds, subj[i], side[i])
        V = np.asarray(m.vertices); N = np.asarray(m.vertex_normals)
        _, basis = ear_frame(pred[i], V)
        n_out = basis[2]
        tree = cKDTree(V)
        rays = pred[i][idx][:, None, :] - ts[None, :, None] * n_out[None, None, :]   # (85, T, 3)
        d, j = tree.query(rays.reshape(-1, 3))
        d = d.reshape(len(idx), len(ts)); j = j.reshape(len(idx), len(ts))
        facing = (N[j] @ n_out) > 0.3
        hit = (d < 0.8) & facing
        first = np.where(hit.any(1), hit.argmax(1), -1)
        prof[i] = np.where(first >= 0, ts[np.maximum(first, 0)], 40.0)
        # auriculocephalic-angle proxy: ear-plane normal vs mean scalp normal behind the ear
        hits = j[hit]
        if len(hits):
            sn = N[hits].mean(0); sn /= np.linalg.norm(sn) + 1e-12
            angle[i] = np.degrees(np.arccos(np.clip(sn @ n_out, -1, 1)))
        del m, V, N, tree; gc.collect()
        if (i + 1) % 50 == 0:
            print(f"  head profiles {i+1}/{n}", flush=True)
    np.savez_compressed(cache, keys=keys, prof=prof, angle=angle)
    return prof, angle


def nested_extra(pred, truth, subj, fold_of, extra):
    """Fully nested per-anchor ridge; extra=None gives V1."""
    PA, TA = pred[:, AI], truth[:, AI]
    final = PA.copy()

    def model(tr_i, te_i):
        out = np.zeros((len(te_i), len(AI), 3))
        for k in range(len(AI)):
            def X(ii):
                rel = (PA[ii] - PA[ii, k][:, None, :]).reshape(len(ii), -1)
                return rel if extra is None else np.hstack([rel, extra[ii]])
            out[:, k] = PA[te_i, k] + fit_predict(X(tr_i), TA[tr_i, k] - PA[tr_i, k], subj[tr_i], X(te_i))
        return out

    for f in range(3):
        tr = np.flatnonzero(fold_of != f); te = np.flatnonzero(fold_of == f)
        us = np.array(sorted(set(subj[tr]))); np.random.default_rng(f).shuffle(us)
        inner = [tr[np.isin(subj[tr], p)] for p in np.array_split(us, 2)]
        oof = {}
        for j in range(2):
            b = inner[j]; a = inner[1 - j]
            for ii, v in zip(b, model(a, b)):
                oof[ii] = v
        step = np.array([oof[ii] for ii in tr]) - PA[tr]
        w = np.array([STEPS[int(np.argmin([np.linalg.norm(PA[tr, k] + s * step[:, k] - TA[tr, k], axis=1).mean()
                                            for s in STEPS]))] for k in range(len(AI))])
        final[te] = PA[te] + w[None, :, None] * (model(tr, te) - PA[te])
    return final


def main():
    pred, truth, subj, side, keys = load_sorted("twopass_surfsnap.npz")
    n = len(pred)
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    fold_of = np.zeros(n, int)
    for f, (_, te) in enumerate(k_fold_subject_split(ds.subject_ids[:200], 3, 0)):
        fold_of[np.isin(subj, list(te))] = f

    print("measuring helix-to-head distance profiles on predicted contours...", flush=True)
    prof, angle = head_profiles(pred, subj, side, keys)
    oh = prof[:, 0:25]
    thirds = [oh[:, 0:8].mean(), oh[:, 8:17].mean(), oh[:, 17:25].mean()]
    print(f"  outer helix head distance by third (mm): upper {thirds[0]:.1f}  middle {thirds[1]:.1f}  "
          f"lower {thirds[2]:.1f}   (otoplasty norm 10-12 / 16-18 / 20-22)")
    print(f"  hit rate: {(prof < 40).mean()*100:.1f}% of rays reached the scalp; "
          f"angle proxy {angle.mean():.1f} +- {angle.std():.1f} deg")

    base = np.linalg.norm(pred - truth, axis=2)
    variants = [("V1 ridge (nested)", None),
                ("V1 + outer-helix head profile", oh),
                ("V1 + all-contour head profile", prof),
                ("V1 + profile + angle", np.hstack([prof, angle[:, None]]))]
    print(f"\nbaseline {base.mean():.4f}")
    ref = None
    for name, extra in variants:
        A = nested_extra(pred, truth, subj, fold_of, extra)
        na = pred.copy(); na[:, AI] = A
        e = np.linalg.norm(rebuild(pred, na) - truth, axis=2); ear = e.mean(1)
        ref = ear if ref is None else ref
        d0, d1 = ear - base.mean(1), ear - ref
        print(f"  {name:32s} {e.mean():.4f}  vs base {d0.mean():+.4f} (t {d0.mean()/(d0.std(ddof=1)/np.sqrt(n)):+.2f})"
              f"  vs V1 {d1.mean():+.4f} (t {d1.mean()/(d1.std(ddof=1)/np.sqrt(n)+1e-12):+.2f})"
              f"   0 {e[:, 0].mean():.3f}  6 {e[:, 6].mean():.3f}  22 {e[:, 22].mean():.3f}  24 {e[:, 24].mean():.3f}",
              flush=True)


if __name__ == "__main__":
    main()
