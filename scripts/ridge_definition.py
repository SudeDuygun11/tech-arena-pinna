"""Reverse-engineer what "on the top of the ridge" means in the annotation tool.

Why: the organizer's extremum rules reproduce landmarks 6 and 22 to a median of
0.15 / 0.32mm on the TRUE curve, but fail on any curve we can produce (our
polyline 0.66mm from truth, a naive max-height crest 0.73mm): an extremum sits on
a flat stretch of rim, so ~0.6mm of shape error becomes ~2mm along it. The
ground-truth contour points were placed by a tool that followed SOME precise
surface feature. If we can identify which one, we can trace the contour the way
the tool did, and the rules become usable.

Test, on ground truth only: at every true contour point, sample the scan
surface ACROSS the contour at 0.1mm steps (-4..+4mm) and locate each candidate
feature. A criterion that is the tool's definition lands at offset ~0 with a
small spread.

  maxheight   highest surface point along the local normal
  maxcurv     sharpest bend across the rim (max of -h'')
  silh_ear    outline seen along the ear-plane normal (surface normal
              perpendicular to that view direction)
  silh_lat    outline seen from the side of the head (normal perpendicular to
              the canonical interaural axis)
  facing_ear  most outward-facing point toward the ear-plane normal
  facing_lat  most outward-facing point toward the lateral axis
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import CONTOUR_SPECS
from src.foundations.dataset import Dataset

DATA = ROOT / "2026 Munich Tech Arena - Datas"
OFFS = np.arange(-4.0, 4.001, 0.1)


def main(n_subjects=60):
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    crit = ["maxheight", "maxcurv", "silh_ear", "silh_lat", "facing_ear", "facing_lat"]
    res = {c: {name: [] for name in CONTOUR_SPECS} for c in crit}
    for e in iter_landmarks_only(ds, ds.subject_ids[:n_subjects]):
        L = np.asarray(e.landmarks, float)
        m = load_canonical_mesh(ds, e.subject_id, e.side)
        V = np.asarray(m.vertices, float); N = np.asarray(m.vertex_normals, float); del m; gc.collect()
        tree = cKDTree(V)
        c = L.mean(0); n_out = np.linalg.svd(L - c, full_matrices=False)[2][2]
        if np.dot(n_out, c - V.mean(0)) < 0:
            n_out = -n_out
        lat = np.array([0.0, 1.0, 0.0]) * np.sign(c[1] if abs(c[1]) > 1e-6 else 1.0)
        for name, spec in CONTOUR_SPECS.items():
            s, en = spec["range"]
            for g in range(s + 1, en - 1):                      # interior points: tangent well defined
                p = L[g]; t = L[g + 1] - L[g - 1]; t /= np.linalg.norm(t)
                _, v0 = tree.query(p); n0 = N[v0]
                across = np.cross(t, n0); across /= np.linalg.norm(across) + 1e-12
                Q = p + OFFS[:, None] * across
                _, jq = tree.query(Q)
                h = (V[jq] - p) @ n0
                nq = N[jq]
                if np.ptp(h) < 1e-3:
                    continue
                hs = np.convolve(h, np.ones(5) / 5, mode="same")
                curv = -np.gradient(np.gradient(hs, 0.1), 0.1)
                a_ear = nq @ n_out; a_lat = nq @ lat
                mid = len(OFFS) // 2

                def zero_nearest(a):
                    sgn = np.sign(a); idx = np.flatnonzero(sgn[:-1] * sgn[1:] < 0)
                    if not len(idx):
                        return np.nan
                    j = idx[np.argmin(np.abs(idx - mid))]
                    return OFFS[j] + 0.1 * a[j] / (a[j] - a[j + 1])

                sel = slice(5, -5)
                res["maxheight"][name].append(OFFS[sel][np.argmax(hs[sel])])
                res["maxcurv"][name].append(OFFS[sel][np.argmax(curv[sel])])
                res["silh_ear"][name].append(zero_nearest(a_ear))
                res["silh_lat"][name].append(zero_nearest(a_lat))
                res["facing_ear"][name].append(OFFS[sel][np.argmax(a_ear[sel])])
                res["facing_lat"][name].append(OFFS[sel][np.argmax(a_lat[sel])])
        del V, N, tree; gc.collect()

    print(f"offset (mm) of each candidate feature from the TRUE contour point, across the rim "
          f"({n_subjects} subjects, both ears)\n  a criterion that defines the contour sits at mean ~0 with small spread\n")
    for name in CONTOUR_SPECS:
        print(f"{name}:")
        for cr in crit:
            v = np.array(res[cr][name], float); v = v[np.isfinite(v)]
            if not len(v):
                continue
            print(f"   {cr:11s} mean {v.mean():+6.2f}  sd {v.std():5.2f}  |offset| {np.abs(v).mean():5.2f}  "
                  f"within 0.5mm {100*(np.abs(v) <= 0.5).mean():5.1f}%   (n={len(v)})")


if __name__ == "__main__":
    main()
