"""Single source of truth for the 85-point pinna landmark layout.

Each ear has 4 contours. Every contour has a small number of anatomically
defined *anchor* landmarks; all other landmarks on that contour are
equally (chord-length) spaced interpolants between consecutive anchors.
See ``pinna_project_documentation.docx`` for the anatomical definitions.

ROLE IN THE PIPELINE
--------------------
Reading order : 3 of 20   (foundations)
Duty          : Single source of truth for the 85-point layout.

Defines the four contours, their index ranges, which 15 indices are anchors,
and the anchor-to-anchor segment iteration used by interpolation. Every
other module derives its indexing from here.
"""
from __future__ import annotations

from dataclasses import dataclass

N_LANDMARKS = 85

# name -> (start, end) half-open global index range, anchor global indices
CONTOUR_SPECS = {
    "outer_helix": {"range": (0, 25), "anchors": [0, 6, 22, 24]},
    "concha_outline": {"range": (25, 55), "anchors": [25, 33, 42, 46, 50, 54]},
    "inner_helix": {"range": (55, 75), "anchors": [55, 64, 74]},
    "superior_antihelix": {"range": (75, 85), "anchors": [75, 84]},
}

CONTOUR_ORDER = ["outer_helix", "concha_outline", "inner_helix", "superior_antihelix"]

# flattened, sorted list of all anchor global indices (15 total)
ANCHOR_INDICES = sorted(
    idx for spec in CONTOUR_SPECS.values() for idx in spec["anchors"]
)
N_ANCHORS = len(ANCHOR_INDICES)
ANCHOR_POSITION = {idx: pos for pos, idx in enumerate(ANCHOR_INDICES)}

# per-contour local anchor indexing, for training separate per-contour models
CONTOUR_ANCHOR_LOCAL_POSITION = {
    name: {idx: pos for pos, idx in enumerate(spec["anchors"])}
    for name, spec in CONTOUR_SPECS.items()
}
CONTOUR_ANCHOR_COUNT = {name: len(spec["anchors"]) for name, spec in CONTOUR_SPECS.items()}


@dataclass(frozen=True)
class ContourSegment:
    """One anchor-to-anchor sub-segment of a contour (inclusive of both ends)."""

    contour: str
    start_anchor: int  # global landmark index of first anchor
    end_anchor: int  # global landmark index of last anchor
    global_indices: tuple  # global indices of all landmarks in this segment, incl. anchors


def iter_segments():
    """Yield every anchor-to-anchor ContourSegment across all 4 contours."""
    for name, spec in CONTOUR_SPECS.items():
        anchors = spec["anchors"]
        for a0, a1 in zip(anchors[:-1], anchors[1:]):
            indices = tuple(range(a0, a1 + 1))
            yield ContourSegment(contour=name, start_anchor=a0, end_anchor=a1, global_indices=indices)


def contour_of(global_index: int) -> str:
    for name, spec in CONTOUR_SPECS.items():
        start, end = spec["range"]
        if start <= global_index < end:
            return name
    raise ValueError(f"index {global_index} out of range")
