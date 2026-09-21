"""Recover the viewing direction the inner helix was annotated in.

ridge_definition.py found: outer helix and concha follow the ridge top (95% /
90% of true points within 0.5mm), but the INNER helix follows an OUTLINE -- the
place where the surface turns away from a viewer (normal perpendicular to the
view direction): 54% within 0.5mm, sd 0.89mm in the canonical side view. That
looser spread suggests the side view is close to, but not exactly, the camera
the annotator used. The organizer's slide also shows 64 directly below 6 on a
vertical line IN THE PICTURE.

Search: view directions on a spherical cap around the lateral axis and around
the ear-plane normal (both tested, in the head frame and in the per-ear frame).
For each, the across-rim offset of the outline from every true inner-helix
point; the best direction minimises the spread. Then, in that view, are 6 and
64 in the same image column?
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
from src.foundations.dataset import Dataset

DATA = ROOT / "2026 Munich Tech Arena - Datas"
OFFS = np.arange(-4.0, 4.001, 0.1)


def cap(center, max_deg=40, step_deg=4):
    center = center / np.linalg.norm(center)
    a = np.array([1.0, 0, 0]) if abs(center[0]) < 0.9 else np.array([0, 0, 1.0])
    u = np.cross(center, a); u /= np.linalg.norm(u); w = np.cross(center, u)
    dirs, ang = [], []
    for th in np.arange(0, max_deg + 1e-9, step_deg):
        for ph in (np.arange(0, 360, max(step_deg, 360 / max(1, int(2 * np.pi * np.sin(np.radians(th)) / np.radians(step_deg))))) if th > 0 else [0]):
            t, p = np.radians(th), np.radians(ph)
            dirs.append(np.cos(t) * center + np.sin(t) * (np.cos(p) * u + np.sin(p) * w)); ang.append((th, ph))
    return np.array(dirs), ang


def main(n_subjects=60):
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    NQ, EAR_B, LM = [], [], []
    for e in iter_landmarks_only(ds, ds.subject_ids[:n_subjects]):
        L = np.asarray(e.landmarks, float)
        m = load_canonical_mesh(ds, e.subject_id, e.side)
        V = np.asarray(m.vertices, float); N = np.asarray(m.vertex_normals, float); del m; gc.collect()
        tree = cKDTree(V)
        c = L.mean(0); B = np.linalg.svd(L - c, full_matrices=False)[2]
        if np.dot(B[2], c - V.mean(0)) < 0:
            B[2] = -B[2]
        if B[0, 2] < 0:
            B[0] = -B[0]
        B[1] = np.cross(B[2], B[0])
        rows = []
        for g in range(56, 74):
            p = L[g]; t = L[g + 1] - L[g - 1]; t /= np.linalg.norm(t)
            _, v0 = tree.query(p); n0 = N[v0]
            ac = np.cross(t, n0); ac /= np.linalg.norm(ac) + 1e-12
            _, jq = tree.query(p + OFFS[:, None] * ac)
            rows.append(N[jq])
        NQ.append(np.array(rows)); EAR_B.append(B); LM.append(L)
        del V, N, tree; gc.collect()
    NQ = np.array(NQ)                          # (ears, 18 points, 81 offsets, 3)
    mid = len(OFFS) // 2

    def spread_for(dirs_per_ear, chunk=40):
        """Chunked over directions to keep peak memory low."""
        parts = [spread_chunk(dirs_per_ear[:, k:k + chunk]) for k in range(0, dirs_per_ear.shape[1], chunk)]
        return tuple(np.concatenate([p[j] for p in parts]) for j in range(3))

    def spread_chunk(dirs_per_ear):
        """dirs_per_ear: (ears, D, 3) -> per-direction sd and share within 0.5mm."""
        a = np.einsum("epoc,edc->epod", NQ, dirs_per_ear)            # (ears, pts, offs, D)
        s = np.sign(a); cross = s[:, :, :-1, :] * s[:, :, 1:, :] < 0
        idx = np.arange(len(OFFS) - 1)[None, None, :, None]
        dist = np.where(cross, np.abs(idx - mid), 10 ** 6)
        j = dist.argmin(2)                                            # nearest crossing
        has = np.take_along_axis(cross, j[:, :, None, :], 2)[:, :, 0, :]
        off = OFFS[j]
        off = np.where(has, off, np.nan).reshape(-1, off.shape[-1])
        sd = np.nanstd(off, 0); within = np.nanmean(np.abs(off) <= 0.5, 0); cover = np.mean(np.isfinite(off), 0)
        return sd, within, cover

    results = []
    lat = np.array([0, 1.0, 0])
    for label, frame in (("head frame, around lateral axis", "head_lat"), ("head frame, around -lateral axis", "head_nlat"),
                         ("ear frame, around ear normal", "ear")):
        if frame == "ear":
            local, ang = cap(np.array([0, 0, 1.0]))
            dirs = np.einsum("dk,ekc->edc", local, np.array(EAR_B))
        else:
            ctr = lat if frame == "head_lat" else -lat
            d0, ang = cap(ctr); dirs = np.repeat(d0[None], len(NQ), 0)
        sd, within, cover = spread_for(dirs)
        score = np.where(cover > 0.8, within, -1)
        b = int(np.argmax(score))
        results.append((within[b], label, ang[b], sd[b], cover[b], dirs[:, b]))
        print(f"{label:34s} best: tilt {ang[b][0]:4.0f} deg, azimuth {ang[b][1]:5.0f}   within 0.5mm {100*within[b]:5.1f}%  "
              f"sd {sd[b]:.2f}mm  coverage {100*cover[b]:.0f}%   (untilted: {100*within[0]:.1f}%, sd {sd[0]:.2f})", flush=True)

    best = max(results, key=lambda r: r[0])
    print(f"\nbest overall: {best[1]}, tilt {best[2][0]} deg -> {100*best[0]:.1f}% of inner-helix points within 0.5mm of the outline")
    v = best[5]
    L = np.array(LM)
    sep_h, sep_v = [], []
    for i in range(len(L)):
        up = np.array([0, 0, 1.0]) - v[i] * v[i, 2]; up /= np.linalg.norm(up)
        right = np.cross(up, v[i])
        d = L[i, 64] - L[i, 6]
        sep_h.append(d @ right); sep_v.append(d @ up)
    sep_h, sep_v = np.array(sep_h), np.array(sep_v)
    print(f"in that view, 64 relative to 6:  horizontal {sep_h.mean():+.2f} +- {sep_h.std():.2f} mm   "
          f"vertical {sep_v.mean():+.2f} +- {sep_v.std():.2f} mm")
    print("  (a small horizontal spread = 64 is directly below 6 in the annotation image, as on the organizer slide)")


if __name__ == "__main__":
    main()
