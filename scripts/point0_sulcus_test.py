"""Test one hypothesis about landmark 0, on real meshes, with no human in the loop.

THE HYPOTHESIS (from manual inspection of the ridge-tinted viewer, 24 ears)
--------------------------------------------------------------------------
Landmark 0 is "upper connection of helix with head, at the center of the ridge".
Walking the helix crest from landmark 6 towards the head, the surface goes:

    convex crest  ->  flat  ->  mildly concave  ->  deep concave (the sulcus)

and landmark 0 sits at the ONSET of concavity -- not at the crest, and not at
the deepest part of the groove. The deep groove is the auriculocephalic sulcus,
where the ear meets the skull; the helix stops being a free-standing ridge
exactly there.

If true this is a threshold-free rule, which is what makes it interesting: the
onset of concavity is a SIGN CHANGE, not a cut-off on a fading quantity. The
organisers stated there is no deterministic rule for landmark 0, so the bar
here is empirical -- does the sign change land where the annotator did?

WHAT IS MEASURED
----------------
Stage A (assumption-free): the smoothed curvature AT ground-truth landmark 0,
compared with landmark 6 and with the intervening landmarks 1-5. If the
hypothesis holds, curvature at 0 should sit near zero with low spread, while 6
should be clearly convex. This tests the SIGNATURE without needing to predict
anything, so it cannot be confounded by a bad walk.

Stage B (the actual rule): walk the crest from landmark 6 and stop at the first
convex->concave sign change. Report the distance to ground-truth landmark 0.

WHAT THIS TEST IS AND IS NOT
----------------------------
It is given ground-truth landmark 6 as the starting point, and ground-truth
landmark 22 to choose which way along the crest to walk (the crest runs both
ways from 6; 22 is 16 indices away and lies in the opposite direction, so this
picks a direction without hinting where 0 is). It therefore measures the
STOPPING RULE, not end-to-end prediction of landmark 0 from a bare mesh. If the
stopping rule works, the starting point can be replaced by the model's own
landmark 6 afterwards -- that error would compound and is not measured here.

CURVATURE CONVENTION
--------------------
    c(v) = ( mean(neighbours of v) - v ) . n(v)
Negative = convex (the neighbours sit below v along the normal: a ridge crest).
Positive = concave (a valley). This is a discrete Laplacian projected on the
normal, i.e. proportional to mean curvature times edge length. It is the same
quantity the viewer tints amber/teal, so the numbers here correspond directly
to what was seen by eye.

Smoothing radius is the one free parameter and is swept, because discrete
curvature on a raw scan is noisy and sign changes are its noisiest feature --
an eye smooths automatically and at the right scale, and that is the most
likely way this works visually but fails in code. If the answer moves a lot
with the radius, the result is an artefact of the setting rather than anatomy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.dataset import Dataset

P0, P6, P22 = 0, 6, 22


def concavity(verts, faces, normals):
    """Per-vertex ( mean(neighbours) - v ) . n. Negative convex, positive concave."""
    acc = np.zeros_like(verts)
    cnt = np.zeros(len(verts))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        i, j = faces[:, a], faces[:, b]
        np.add.at(acc, i, verts[j]);  np.add.at(cnt, i, 1)
        np.add.at(acc, j, verts[i]);  np.add.at(cnt, j, 1)
    cnt = np.maximum(cnt, 1)[:, None]
    return np.einsum("ij,ij->i", acc / cnt - verts, normals)


def smooth_field(tree, verts, field, radius):
    """Average the field over a Euclidean ball. Euclidean (not geodesic) is a
    deliberate simplification and a known risk: across a thin helix rim the ball
    can reach the far side of the ridge. Radii are kept at or below the rim
    half-thickness to limit that."""
    out = np.empty_like(field)
    for i, idx in enumerate(tree.query_ball_point(verts, radius)):
        out[i] = field[idx].mean() if len(idx) else field[i]
    return out


def walk_crest(verts, tree, curv, start, direction, step=0.8, max_mm=36.0,
               recentre_r=1.6, momentum=0.75):
    """Walk along the ridge from `start` in `direction`, staying on the crest.

    Each step: advance `step` mm, snap to the nearest surface vertex, then
    re-centre onto the crest by taking the convexity-weighted centroid of nearby
    vertices. Re-centring is what keeps the walk on the middle of the ridge
    rather than sliding down a slope -- the "centre of the ridge across its
    thickness" the annotation protocol asks for.

    Returns (positions, curvature_along_walk).
    """
    p = np.asarray(start, dtype=float)
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d) + 1e-12

    pos, cs, travelled = [p.copy()], [curv[tree.query(p)[1]]], 0.0
    while travelled < max_mm:
        q = p + step * d
        q = verts[tree.query(q)[1]]                       # back onto the surface

        idx = tree.query_ball_point(q, recentre_r)
        if idx:
            # Weight the most CONVEX neighbours highest -> pulls onto the crest.
            # The correction is projected PERPENDICULAR to the heading: applied
            # in full it simply pulls back to the same local convexity maximum
            # every step, and the walk converges to a fixed point after ~2mm.
            w = np.exp(-(curv[idx] - curv[idx].min()) / 0.6)
            sw = w.sum()
            if sw > 1e-9:
                centre = (verts[idx] * w[:, None]).sum(axis=0) / sw
                corr = centre - q
                corr -= np.dot(corr, d) * d               # keep the forward motion
                q = verts[tree.query(q + corr)[1]]

        nd = q - p
        n = np.linalg.norm(nd)
        if n < 1e-6:
            break
        nd /= n
        d = momentum * d + (1 - momentum) * nd            # smooth the heading
        d /= np.linalg.norm(d) + 1e-12

        travelled += n
        p = q
        pos.append(p.copy())
        cs.append(curv[tree.query(p)[1]])
    return np.array(pos), np.array(cs)


def zero_crossing_points(verts, faces, curv):
    """Points where smoothed curvature changes sign, located on mesh EDGES.

    For every edge whose endpoints straddle zero, the crossing is placed by
    linear interpolation along that edge. This is the discrete parabolic line --
    the boundary between convex and concave surface, i.e. the amber/teal
    frontier in the viewer.
    """
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ci, cj = curv[e[:, 0]], curv[e[:, 1]]
    m = (ci < 0) != (cj < 0)
    if not m.any():
        return np.empty((0, 3))
    a, b = e[m, 0], e[m, 1]
    t = (ci[m] / (ci[m] - cj[m]))[:, None]
    return verts[a] + t * (verts[b] - verts[a])


def ridge_tangent(verts, tree, curv, p, radius=5.0, convex_frac=0.5):
    """Local direction of the ridge at p, from the mesh alone.

    Takes the most convex vertices in a ball around p -- these lie along the
    crest line, since a ridge is convex along its top and falls away to either
    side -- and returns their first principal axis. Sign is arbitrary; both
    directions are walked by the caller.
    """
    idx = np.asarray(tree.query_ball_point(p, radius))
    if len(idx) < 8:
        return None
    keep = idx[curv[idx] <= np.quantile(curv[idx], convex_frac)]
    if len(keep) < 6:
        return None
    X = verts[keep] - verts[keep].mean(axis=0)
    return np.linalg.svd(X, full_matrices=False)[2][0]


def first_sign_change(pos, cs, need_convex_first=True):
    """First convex(-) -> concave(+) crossing, linearly interpolated between the
    two bracketing samples so the answer is not quantised to the vertex spacing
    (mesh edges are ~0.8mm, and we are chasing tenths of a millimetre)."""
    start = 0
    if need_convex_first:
        conv = np.flatnonzero(cs < 0)
        if len(conv) == 0:
            return None
        start = conv[0]
    for i in range(start, len(cs) - 1):
        if cs[i] < 0 <= cs[i + 1]:
            t = cs[i] / (cs[i] - cs[i + 1])               # in [0,1]
            return pos[i] + t * (pos[i + 1] - pos[i])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-ears", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radii", type=float, nargs="+", default=[1.0, 2.0, 3.5])
    ap.add_argument("--crop", type=float, default=45.0)
    args = ap.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(ds.subject_ids), size=min(args.n_ears, len(ds)), replace=False)

    statA = {r: {"p0": [], "p6": [], "p1_5": [],
             "d0": [], "d1_5": [], "drand": []} for r in args.radii}
    statB = {r: {"err": [], "fail": 0} for r in args.radii}
    gap = []

    for n, si in enumerate(picks, 1):
        sid = ds.subject_ids[si]
        try:
            mesh, lm_left, _ = ds[si]
        except Exception as exc:
            print(f"[{n:3d}/{len(picks)}] {sid}: load failed ({exc})")
            continue
        L = np.asarray(lm_left, dtype=float)

        # crop to the ear neighbourhood, then release the full head immediately
        centre = L[P6]
        v_all = np.asarray(mesh.vertices)
        keep_v = np.linalg.norm(v_all - centre, axis=1) <= args.crop
        fmask = keep_v[np.asarray(mesh.faces)].all(axis=1)
        sub = mesh.submesh([np.flatnonzero(fmask)], append=True)
        verts = np.asarray(sub.vertices, dtype=float)
        faces = np.asarray(sub.faces)
        norms = np.asarray(sub.vertex_normals, dtype=float)
        del mesh, sub, v_all

        tree = cKDTree(verts)
        raw = concavity(verts, faces, norms)
        # control population: surface within 20mm of landmark 6
        near_idx = np.flatnonzero(np.linalg.norm(verts - L[P6], axis=1) <= 20.0)
        gap.append(np.linalg.norm(L[P0] - L[P6]))

        per_ear = {}
        for r in args.radii:
            cur = smooth_field(tree, verts, raw, r)
            cur = cur / (np.abs(cur).mean() + 1e-12)      # per-ear scale normalise

            # ---- Stage A: curvature signature at the ground-truth landmarks
            statA[r]["p0"].append(cur[tree.query(L[P0])[1]])
            statA[r]["p6"].append(cur[tree.query(L[P6])[1]])
            statA[r]["p1_5"].append(np.mean([cur[tree.query(L[k])[1]] for k in range(1, 6)]))

            # Distance from ground truth to the nearest zero crossing, WITH
            # CONTROLS. On its own "landmark 0 is 0.2mm from a crossing" proves
            # nothing -- if crossings are dense everywhere then every point is
            # near one. Landmarks 1-5 (which should be firmly convex) and random
            # surface points near the ear are the controls: the cue only carries
            # information if landmark 0 is much closer than they are.
            zc = zero_crossing_points(verts, faces, cur)
            if len(zc):
                zt = cKDTree(zc)
                statA[r]["d0"].append(float(zt.query(L[P0])[0]))
                statA[r]["d1_5"].append(float(np.mean([zt.query(L[k])[0] for k in range(1, 6)])))
                sel = rng.choice(len(near_idx), size=min(200, len(near_idx)), replace=False)
                statA[r]["drand"].append(float(np.mean(zt.query(verts[near_idx[sel]])[0])))

            # ---- Stage B: walk BOTH ways along the ridge from landmark 6.
            # The chord towards 22 is 122.8 deg away from the true heading on
            # average, so it cannot disambiguate. Anatomy can: landmark 0 is the
            # near end of the crest (~20mm from 6), while the other direction
            # runs past 22 and on to 24, roughly 90mm away. So take whichever
            # direction reaches a convex->concave crossing FIRST. No ground
            # truth is used to make this choice.
            t = ridge_tangent(verts, tree, cur, L[P6])
            hit, best = None, np.inf
            if t is not None:
                for sgn in (1.0, -1.0):
                    pos, cs = walk_crest(verts, tree, cur, L[P6], sgn * t)
                    h = first_sign_change(pos, cs)
                    if h is None:
                        continue
                    travel = np.linalg.norm(h - L[P6])
                    if travel < best:
                        best, hit = travel, h
            if hit is None:
                statB[r]["fail"] += 1
                per_ear[r] = "  FAIL"
            else:
                e = float(np.linalg.norm(hit - L[P0]))
                statB[r]["err"].append(e)
                per_ear[r] = f"{e:6.2f}"

        print(f"[{n:3d}/{len(picks)}] {sid}  " +
              "  ".join(f"r={r}:{per_ear[r]}" for r in args.radii), flush=True)

    print("\n" + "=" * 74)
    print(f"n ears = {len(gap)}    |0-6| distance: mean {np.mean(gap):.2f}mm "
          f"(this is what 'predict nothing, return landmark 6' would cost)")

    print("\nSTAGE A -- smoothed curvature at ground truth (normalised; <0 convex, >0 concave)")
    print(f"{'radius':>7s} {'at lm 0':>18s} {'at lm 6':>18s} {'at lm 1-5':>18s}")
    for r in args.radii:
        f = lambda k: (np.mean(statA[r][k]), np.std(statA[r][k]))
        m0, s0 = f("p0"); m6, s6 = f("p6"); m1, s1 = f("p1_5")
        print(f"{r:7.1f} {m0:9.3f}+-{s0:<7.3f} {m6:9.3f}+-{s6:<7.3f} {m1:9.3f}+-{s1:<7.3f}")
    print("  hypothesis predicts: lm 0 near 0, lm 6 clearly negative, lm 1-5 negative")

    print()
    print("STAGE A2 -- distance (mm) to the nearest curvature zero crossing")
    print(f"{'radius':>7s} {'lm 0':>16s} {'lm 1-5 (ctrl)':>18s} {'random pts (ctrl)':>19s}")
    for r in args.radii:
        g = lambda k: (np.mean(statA[r][k]), np.std(statA[r][k])) if statA[r][k] else (float('nan'),)*2
        a0, s0 = g("d0"); a1, s1 = g("d1_5"); ar, sr = g("drand")
        print(f"{r:7.1f} {a0:8.3f}+-{s0:<6.3f} {a1:9.3f}+-{s1:<7.3f} {ar:10.3f}+-{sr:<7.3f}")
    print("  the cue only carries information if lm 0 is MUCH closer than the controls")

    print("\nSTAGE B -- crest walk from lm 6, stop at first convex->concave crossing")
    print(f"{'radius':>7s} {'n':>4s} {'fail':>5s} {'mean':>8s} {'median':>8s} "
          f"{'p90':>8s} {'best':>8s} {'worst':>8s}")
    for r in args.radii:
        e = np.array(statB[r]["err"])
        if len(e) == 0:
            print(f"{r:7.1f} {0:4d} {statB[r]['fail']:5d}      all walks failed")
            continue
        print(f"{r:7.1f} {len(e):4d} {statB[r]['fail']:5d} {e.mean():8.3f} "
              f"{np.median(e):8.3f} {np.percentile(e,90):8.3f} {e.min():8.3f} {e.max():8.3f}")
    print("\nreference: our model's outer_helix error is 2.09mm (all 25 points, held out)")


if __name__ == "__main__":
    main()
