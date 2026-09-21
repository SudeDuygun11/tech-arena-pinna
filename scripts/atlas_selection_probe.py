"""Is the registration prior limited by WHICH reference ears are used?

The along-curve position of every landmark comes, before any network, from the
registration prior: the mean of 11 warped reference ears. Those 11 are the same
for every test ear -- the 11 most TYPICAL training ears (closest to the mean
shape). k 7 -> 11 was one of the largest single wins, and k > 11 was never tried;
nor was choosing references per test ear (standard multi-atlas practice).

Fold 0 (133 train subjects -> 134 held-out ears), raw registration only (no
network): register every held-out ear to the 40 most typical references, cache
the 40 individual predictions, then score fusion rules:
  * mean of first k (k = 5, 11, 20, 30, 40)        -- more atlases
  * median / trimmed mean of 40                    -- robust fusion
  * consensus selection: per ear keep the 11 references whose predicted shape is
    closest to the 40-median (no truth used)        -- per-ear selection
  * ORACLE: per ear, the 11 references with the lowest true error (ceiling)
Anchors and all 85 points reported, plus the along/across split for anchors.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.foundations.canonical import iter_landmarks_only, load_canonical_mesh
from src.foundations.contours import ANCHOR_INDICES
from src.foundations.dataset import Dataset
from src.foundations.splits import k_fold_subject_split
from src.registration.registration import register
from src.registration.template import build_template

DATA = ROOT / "2026 Munich Tech Arena - Datas"
CACHE = ROOT / "results" / "atlas_k40_fold0.npz"
K_MAX = 40
AI = list(ANCHOR_INDICES)


def main():
    ds = Dataset(mesh_dir=str(DATA / "mesh"), landmarks_dir=str(DATA / "landmarks"))
    train, test = next(iter(k_fold_subject_split(ds.subject_ids[:200], 3, 0)))
    ears = [(s, side) for s in test for side in ("left", "right")]
    truth = {}
    for ex in iter_landmarks_only(ds, list(test)):
        truth[(ex.subject_id, ex.side)] = ex.landmarks
    T = np.array([truth[e] for e in ears])

    if CACHE.exists():
        preds = np.load(CACHE)["preds"]
    else:
        t0 = time.time()
        template = build_template(ds, list(train), k_references=K_MAX)
        print(f"template with {len(template.references)} references built in {time.time()-t0:.0f}s", flush=True)
        preds = np.zeros((len(ears), K_MAX, 85, 3))
        for i, (s, side) in enumerate(ears):
            V = np.asarray(load_canonical_mesh(ds, s, side).vertices)
            for r, ref in enumerate(template.references):
                preds[i, r] = register(ref, V, template.global_crop_center, template.global_crop_radius)
            del V; gc.collect()
            if (i + 1) % 10 == 0:
                print(f"  registered {i+1}/{len(ears)} ears ({(time.time()-t0)/60:.1f} min)", flush=True)
        np.savez_compressed(CACHE, preds=preds, ears=np.array([f"{s}_{d}" for s, d in ears]))

    err = lambda P: np.linalg.norm(P - T, axis=2)
    def report(label, P, ref=None):
        e = err(P); line = f"{label:34s} all {e.mean():.4f}  anchors {e[:, AI].mean():.4f}"
        if ref is not None:
            d = e.mean(1) - ref
            line += f"   vs k=11 {d.mean():+.4f} (t {d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.1f})"
        print(line)
        return e.mean(1)

    base = report("mean of first 11 (current)", preds[:, :11].mean(1))
    for k in (5, 20, 30, 40):
        report(f"mean of first {k}", preds[:, :k].mean(1), base)
    report("median of 40", np.median(preds, axis=1), base)
    srt = np.sort(np.linalg.norm(preds - np.median(preds, axis=1, keepdims=True), axis=3), axis=1)
    med = np.median(preds, axis=1, keepdims=True)
    dev = np.linalg.norm(preds - med, axis=3).mean(2)                   # (ears, refs)
    keep = np.argsort(dev, axis=1)[:, :11]
    report("consensus-selected 11 of 40", np.take_along_axis(preds, keep[:, :, None, None], 1).mean(1), base)
    keep20 = np.argsort(dev, axis=1)[:, :20]
    report("consensus-selected 20 of 40", np.take_along_axis(preds, keep20[:, :, None, None], 1).mean(1), base)
    true_err = err_refs = np.linalg.norm(preds - T[:, None], axis=3).mean(2)
    best = np.argsort(true_err, axis=1)[:, :11]
    report("ORACLE best 11 of 40 per ear", np.take_along_axis(preds, best[:, :, None, None], 1).mean(1), base)
    print(f"\nsingle-reference error: mean {true_err.mean():.3f}; best reference per ear {true_err.min(1).mean():.3f}")


if __name__ == "__main__":
    main()
