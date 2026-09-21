"""Full 400-ear check: does re-snapping the current-best predictions onto the
mesh SURFACE (instead of the nearest VERTEX) reduce error?

Post-processing only, on saved predictions from the 1.3736mm run -- no
retraining needed. This is the same style of check as the equidistant
resampling test: if it wins here, it's a candidate to wire into blend_correct
and confirm end-to-end; if it doesn't survive contact with all 400 ears it's
cheaper to find out now than after a 5-hour retrain.

15-ear quick sample already showed 1.2947 -> 1.2777mm (-0.0170). This runs
all 400.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.dataset import Dataset
from src.foundations.canonical import load_canonical_mesh
from src.foundations.geometry import nearest_surface_points


def main():
    d = np.load("results/tier2_trim0_iters16_kref11.npz", allow_pickle=True)
    pred, truth, sids, sides = d["pred"], d["truth"], d["subject_id"], d["side"]
    n = len(pred)
    print(f"{n} ears loaded from the current-best (1.3736mm) run", flush=True)

    ds = Dataset(mesh_dir="2026 Munich Tech Arena - Datas/mesh",
                 landmarks_dir="2026 Munich Tech Arena - Datas/landmarks")

    before = np.zeros(n)
    after = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        sid, side = str(sids[i]), str(sides[i])
        mesh = load_canonical_mesh(ds, sid, side)
        V = np.asarray(mesh.vertices)
        F = np.asarray(mesh.faces)
        p = pred[i]
        surf, _ = nearest_surface_points(V, F, p)
        before[i] = np.linalg.norm(p - truth[i], axis=1).mean()
        after[i] = np.linalg.norm(surf - truth[i], axis=1).mean()
        del mesh
        if (i + 1) % 40 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n}  {el:.0f}s elapsed  ~{el/(i+1)*(n-i-1):.0f}s left  "
                  f"running: before {before[:i+1].mean():.4f}  after {after[:i+1].mean():.4f}",
                  flush=True)

    print("\n" + "=" * 60)
    print(f"vertex-snap (current):  {before.mean():.4f} mm")
    print(f"surface-snap (test):    {after.mean():.4f} mm")
    print(f"delta:                  {after.mean()-before.mean():+.4f} mm")
    print(f"ears improved:          {int((after < before).sum())}/{n}")


if __name__ == "__main__":
    main()
