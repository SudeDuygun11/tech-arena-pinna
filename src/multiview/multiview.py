"""Render an ear from several viewpoints, the way the annotator inspected it.

WHY
---
The annotation protocol described by the organisers is explicitly multi-view:
align to the interaural axis, rotate to roughly 85-95 degrees for the lateral
view, place the landmark, then verify from a superior/top view that it sits
centrally across the ridge rather than on a slope. They also named the exact
failure this guards against: "a point may appear correctly positioned from one
viewing direction while actually lying towards the inner or outer side of the
ridge when inspected from another".

That is our wrong-ridge problem stated by the people who made the data. Every
attempt to solve it from local 3D evidence has failed the same way -- an oracle
over candidates reaches 0.5-0.7mm while an actual selector gets 5.5-6.7mm. The
information is present; choosing among instances of it is what fails. Two views
disagreeing is a selection signal that no single patch can provide.

RENDERING APPROACH
------------------
No GPU rasteriser is available here (pyrender, OpenGL, pytorch3d are all
absent), so this is a pure-numpy point-splat z-buffer:

  * project points, keep the nearest per pixel via np.minimum.at on a flattened
    index -- fully vectorised, no per-triangle Python loop (50k triangles in a
    loop would be seconds per view, and we need ~7000 views)
  * splat VERTICES + FACE CENTROIDS + EDGE MIDPOINTS, which roughly quadruples
    the sample count for free and gives dense coverage without subdivision
  * a small dilation pass fills the residual pinholes

Channels rendered per view: depth, surface normal, and signed smoothed
curvature. The curvature channel is deliberate -- it is the same quantity the
inspection viewer tints amber (convex crest) and teal (concave sulcus), and it
is where landmark 0's signature was found empirically (curvature 0.042 +- 0.665
at the landmark against -2.085 at landmark 6, over 40 ears). Handing the network
the cue directly beats making it rediscover it from depth.

Back-projection data is returned with every view so 2D predictions can be lifted
to 3D: a pixel plus its depth is a ray-length pair, and the camera basis turns
that into a point.


ROLE IN THE PIPELINE
--------------------
Reading order : 19 of 20   (multiview)
Duty          : Renders an ear from N viewpoints, as the annotator inspected it.

Pure-numpy point-splat z-buffer, because no GPU rasteriser is available
here. Splats vertices plus face centroids plus edge midpoints for dense
coverage without subdivision. Orthographic, so millimetres-per-pixel stays
constant across subjects. Channels: depth, normal, signed curvature.

Known issues / status:
  ear_frame must pin BOTH the normal sign and the in-plane axes. SVD signs are
  arbitrary, and a flipped normal puts the camera inside the head: measured on
  P0003_left as 1.2 of 9 views seeing any landmark, against about 6.5 when
  oriented correctly.
"""
from __future__ import annotations

import numpy as np

DEFAULT_RES = 256
DEFAULT_MARGIN = 1.25          # fraction of the landmark extent kept in frame


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Orthonormal camera basis (right, up, forward). Forward points at target."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    up = np.asarray(up, float)
    if abs(np.dot(f, up)) > 0.99:                 # degenerate: pick another up
        up = np.array([1.0, 0.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    return r, u, f


def view_directions(n_lateral: int = 8, superior: bool = True):
    """Unit view directions in the ear's local frame.

    `n_lateral` directions evenly spaced about the ear's own normal axis, plus
    an optional top-down view. Returned in a canonical local frame; the caller
    rotates them into the subject's frame, so the same angles mean the same
    thing across subjects.
    """
    dirs = []
    for k in range(n_lateral):
        a = 2.0 * np.pi * k / n_lateral
        # tilt 25 degrees off the pure normal so views actually differ in
        # parallax; a ring at 0 tilt would all be the same direction
        t = np.radians(25.0)
        dirs.append([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])
    if superior:
        dirs.append([0.0, 1.0, 0.0])
    return np.array(dirs, float)


def ear_frame(landmarks: np.ndarray, mesh_points: np.ndarray | None = None):
    """Local frame from the 85 landmarks: outward normal of their best-fit
    plane, plus two in-plane axes.

    EVERY SIGN HERE MUST BE PINNED DOWN. SVD returns principal axes with
    arbitrary sign, and both failure modes are silent and severe:

      * a flipped NORMAL puts the camera inside the head, so every view renders
        the skull in front of the ear. Measured on P0003_left before this was
        fixed: landmarks lay 0.172mm from the mesh, yet the rendered depth at
        their pixels was 13.152mm out, and only 1.2 of 9 views saw any landmark
        at all (against ~6.5 for correctly-oriented ears).
      * a flipped IN-PLANE axis rotates the whole view ring, so the same view
        index means a different viewing angle on different subjects -- which
        would make the shared 2D backbone see inconsistent poses.

    The normal is oriented away from the bulk of the mesh (the head is inward,
    the pinna outward). The in-plane axes are pinned against the canonical world
    frame, which is consistent across subjects because right ears are mirrored
    into the left-ear frame upstream.
    """
    c = landmarks.mean(axis=0)
    X = landmarks - c
    # rows of Vt are principal axes, last = smallest variance = plane normal
    Vt = np.linalg.svd(X, full_matrices=False)[2]
    e1, nrm = Vt[0], Vt[2]

    if mesh_points is not None and len(mesh_points):
        outward = c - np.asarray(mesh_points).mean(axis=0)
        if np.dot(nrm, outward) < 0:
            nrm = -nrm

    # pin the in-plane axis against a fixed world direction, then rebuild an
    # orthonormal right-handed frame around the (now oriented) normal
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(nrm, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = ref - np.dot(ref, nrm) * nrm
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(nrm, e1)
    return c, np.stack([e1, e2, nrm])


def splat_render(points, normals, curv, eye, target, up, extent,
                 res: int = DEFAULT_RES):
    """Z-buffer splat of a point cloud. Returns depth/normal/curv/mask images.

    `extent` is the half-width of the orthographic frame in mm. Orthographic
    (not perspective) is deliberate: it keeps millimetres-per-pixel constant
    across the image and across subjects, so a 2D heatmap error converts to a
    3D error by one fixed scale factor.
    """
    r, u, f = look_at(eye, target, up)
    rel = points - np.asarray(target, float)
    x = rel @ r
    y = rel @ u
    z = rel @ f                                   # depth along the view axis

    px = ((x / extent) * 0.5 + 0.5) * (res - 1)
    py = (0.5 - (y / extent) * 0.5) * (res - 1)
    ix = np.rint(px).astype(np.int64)
    iy = np.rint(py).astype(np.int64)

    ok = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)
    if not ok.any():
        z_img = np.full((res, res), np.inf)
        return {"depth": z_img, "normal": np.zeros((res, res, 3)),
                "curv": np.zeros((res, res)), "mask": np.zeros((res, res), bool),
                "basis": (r, u, f), "target": np.asarray(target, float),
                "extent": extent, "res": res}

    ix, iy, z = ix[ok], iy[ok], z[ok]
    flat = iy * res + ix

    zbuf = np.full(res * res, np.inf)
    np.minimum.at(zbuf, flat, z)                  # nearest sample wins

    # second pass: keep the attributes of whichever sample won its pixel
    win = np.isclose(z, zbuf[flat])
    nbuf = np.zeros((res * res, 3))
    cbuf = np.zeros(res * res)
    nbuf[flat[win]] = normals[ok][win]
    cbuf[flat[win]] = curv[ok][win]

    depth = zbuf.reshape(res, res)
    mask = np.isfinite(depth)
    return {"depth": depth, "normal": nbuf.reshape(res, res, 3),
            "curv": cbuf.reshape(res, res), "mask": mask,
            "basis": (r, u, f), "target": np.asarray(target, float),
            "extent": extent, "res": res}


def densify(verts, faces, normals, curv):
    """Vertices + face centroids + edge midpoints.

    Splatting vertices alone leaves the image full of pinholes: ~24k vertices
    over a 256x256 frame is under one sample per five pixels. Adding centroids
    and midpoints roughly quadruples the count with no subdivision and no new
    geometry, which is enough for a dense image.
    """
    cen_p = verts[faces].mean(axis=1)
    cen_n = normals[faces].mean(axis=1)
    cen_c = curv[faces].mean(axis=1)

    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    mid_p = verts[e].mean(axis=1)
    mid_n = normals[e].mean(axis=1)
    mid_c = curv[e].mean(axis=1)

    P = np.vstack([verts, cen_p, mid_p])
    N = np.vstack([normals, cen_n, mid_n])
    C = np.concatenate([curv, cen_c, mid_c])
    N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    return P, N, C


def fill_holes(view, iters: int = 2):
    """Fill single-pixel gaps by taking the nearest finite 4-neighbour."""
    d = view["depth"].copy()
    n = view["normal"].copy()
    c = view["curv"].copy()
    for _ in range(iters):
        holes = ~np.isfinite(d)
        if not holes.any():
            break
        best = np.full_like(d, np.inf)
        src = np.zeros(d.shape + (2,), dtype=np.int64)
        rows, cols = np.indices(d.shape)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sr = np.clip(rows + dr, 0, d.shape[0] - 1)
            sc = np.clip(cols + dc, 0, d.shape[1] - 1)
            cand = d[sr, sc]
            take = holes & np.isfinite(cand) & (cand < best)
            best[take] = cand[take]
            src[take] = np.stack([sr, sc], axis=-1)[take]
        got = holes & np.isfinite(best)
        d[got] = best[got]
        n[got] = n[src[got][:, 0], src[got][:, 1]]
        c[got] = c[src[got][:, 0], src[got][:, 1]]
    view = dict(view)
    view["depth"], view["normal"], view["curv"] = d, n, c
    view["mask"] = np.isfinite(d)
    return view


def project(view, pts3d):
    """3D -> (col, row) pixel coordinates, float (sub-pixel)."""
    r, u, f = view["basis"]
    rel = np.atleast_2d(pts3d) - view["target"]
    res, ext = view["res"], view["extent"]
    px = ((rel @ r / ext) * 0.5 + 0.5) * (res - 1)
    py = (0.5 - (rel @ u / ext) * 0.5) * (res - 1)
    return np.stack([px, py], axis=-1)


def backproject(view, px, py, depth):
    """(col, row, depth) -> 3D. Inverse of `project`, used to lift a predicted
    2D heatmap peak back into the subject's coordinate frame."""
    r, u, f = view["basis"]
    res, ext = view["res"], view["extent"]
    x = ((px / (res - 1)) * 2.0 - 1.0) * ext
    y = ((0.5 - py / (res - 1)) * 2.0) * ext
    return view["target"] + x * r + y * u + depth * f


def render_ear(verts, faces, normals, curv, landmarks,
               n_lateral: int = 8, superior: bool = True,
               res: int = DEFAULT_RES, margin: float = DEFAULT_MARGIN):
    """All views of one ear, framed on the landmarks."""
    centre, basis = ear_frame(landmarks, verts)
    extent = np.abs((landmarks - centre) @ basis.T).max() * margin

    P, N, C = densify(verts, faces, normals, curv)
    out = []
    for d_local in view_directions(n_lateral, superior):
        d_world = d_local @ basis                       # local -> subject frame
        eye = centre + d_world * (extent * 6.0)
        up = basis[0]                                   # in-plane axis, stable
        v = fill_holes(splat_render(P, N, C, eye, centre, up, extent, res))
        v["dir_local"] = d_local
        out.append(v)
    return out, centre, basis, extent
