"""Export one ear as a compact, self-contained payload for the manual
annotation tool (annotation_tool.html).

WHY THIS EXISTS
---------------
The slides give the annotator's own definition of each of the 15 anchors, and
most of them are NOT local features -- they are extremal points, junctions
between two contours, references to another landmark, or (index 74) pure
arc-length construction. That predicts a human should be able to place them
from global shape where our 7mm-patch model cannot, and it predicts our error
should track how local a contour's anchors are. It does:

    concha_outline      4 of 6 anchors local   1.2395mm
    superior_antihelix  0 of 2                 1.6019
    inner_helix         0 of 3                 1.9841
    outer_helix         0 of 4                 2.0935

This tool measures the other side of that claim: given the same mesh and the
same written definitions, how close does a HUMAN get? That number is the
annotator-agreement floor -- the thing we have never been able to estimate
because the dataset carries exactly one annotation per ear.

The payload is deliberately small: the ear is cropped from the full head mesh
(15-34MB) and decimated, so the whole thing fits in a browser page.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.contours import ANCHOR_INDICES, CONTOUR_SPECS
from src.foundations.dataset import Dataset

# Verbatim from the challenge slides. These are shown to the annotator in the
# tool -- they ARE the task, so they are quoted exactly rather than paraphrased.
ANCHOR_DEFS = {
    0:  "upper connection of helix with head, at the center of the ridge",
    6:  "upper point of the largest extent of the outer helix, annotated on the top of the ridge",
    22: "lower point of the largest extent of the outer helix, annotated on the top of the ridge",
    24: "lower connection of helix with head, at the center of the ridge",
    25: "connection of concha with helix at 90 degree view",
    33: "junction of fossa and outer concha contour",
    42: "antitragus (at highest curvature)",
    46: "saddle point below tragus",
    50: "tragus (at highest curvature)",
    54: "saddle point above tragus",
    55: "inner helix ridge at height of concha start",
    64: "point opposite the highest point of the outer helix",
    74: "end point of the continuation of the contour line for another 10 points "
        "with the same neighbor distance",
    75: "junction of fossa (crura of antihelix) and outer concha contour",
    84: "connection of fossa (crura of antihelix) with helix",
}

# Which kind of evidence each definition needs. Drives the colour coding in the
# tool, and is the whole point of the exercise: only 4 of 15 are local.
ANCHOR_KIND = {
    42: "local", 46: "local", 50: "local", 54: "local",
    6: "extremal", 22: "extremal",
    25: "junction", 33: "junction", 75: "junction", 84: "junction",
    55: "reference", 64: "reference",
    0: "head", 24: "head",
    74: "constructed",
}


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def crop_ear(mesh: trimesh.Trimesh, landmarks: np.ndarray, margin: float = 15.0):
    """Faces whose centroid lies within (landmark extent + margin) of the ear.

    Cropping is what makes the payload viable at all -- a full head is 15-34MB
    of PLY. Margin is generous on purpose: anchors 0 and 24 are defined as the
    connection of the helix WITH THE HEAD, so cutting tight to the ear would
    remove the very geometry those two are defined against.
    """
    centre = landmarks.mean(axis=0)
    radius = np.linalg.norm(landmarks - centre, axis=1).max() + margin

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    centroids = verts[faces].mean(axis=1)
    keep = np.linalg.norm(centroids - centre, axis=1) <= radius
    sub = mesh.submesh([np.flatnonzero(keep)], append=True)
    return sub, centre, radius


def decimate(mesh: trimesh.Trimesh, target_faces: int):
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception as exc:                       # optional dependency
        print(f"  (decimation unavailable: {exc}; keeping {len(mesh.faces)} faces)")
        return mesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="parent of mesh/ and landmarks/")
    p.add_argument("--subject", default=None, help="e.g. P0001 (default: first subject)")
    p.add_argument("--side", choices=["left", "right"], default="left")
    p.add_argument("--target-faces", type=int, default=40000)
    p.add_argument("--margin", type=float, default=15.0)
    p.add_argument("--out", default="ear_payload.json")
    args = p.parse_args()

    root = Path(args.data_dir)
    ds = Dataset(mesh_dir=str(root / "mesh"), landmarks_dir=str(root / "landmarks"))
    sid = args.subject or ds.subject_ids[0]
    idx = ds.subject_ids.index(sid)

    print(f"loading {sid} ({args.side})...")
    mesh, lm_left, lm_right = ds[idx]
    landmarks = lm_left if args.side == "left" else lm_right

    print(f"  full mesh: {len(mesh.vertices)} verts / {len(mesh.faces)} faces")
    sub, centre, radius = crop_ear(mesh, landmarks, margin=args.margin)
    print(f"  cropped  : {len(sub.vertices)} verts / {len(sub.faces)} faces  (r={radius:.1f}mm)")
    sub = decimate(sub, args.target_faces)
    print(f"  decimated: {len(sub.vertices)} verts / {len(sub.faces)} faces")

    verts = np.asarray(sub.vertices, dtype=np.float64) - centre   # centre for rendering
    faces = np.asarray(sub.faces)
    gt = np.asarray(landmarks, dtype=np.float64) - centre

    if len(verts) >= 65536:
        face_dtype, face_kind = np.uint32, "u32"
    else:
        face_dtype, face_kind = np.uint16, "u16"

    payload = {
        "subject": sid,
        "side": args.side,
        "centre": centre.tolist(),
        "radius": float(radius),
        "n_verts": int(len(verts)),
        "n_faces": int(len(faces)),
        "face_kind": face_kind,
        "verts": _b64(verts, np.float32),
        "faces": _b64(faces, face_dtype),
        # Ground truth ships with the payload so the tool can score offline,
        # and is not revealed in the UI until the user presses Reveal.
        "gt": _b64(gt, np.float32),
        "anchor_indices": list(ANCHOR_INDICES),
        "anchor_defs": {str(k): v for k, v in ANCHOR_DEFS.items()},
        "anchor_kind": {str(k): v for k, v in ANCHOR_KIND.items()},
        "contours": {name: {"range": list(spec["range"]), "anchors": list(spec["anchors"])}
                     for name, spec in CONTOUR_SPECS.items()},
    }

    out = Path(args.out)
    out.write_text(json.dumps(payload))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
