"""Mesh-free control for aligned patches: same examples, world frame vs local frame.

The pipeline screen showed aligned patches much WORSE (seed 0: 1.784 vs ~1.626)
while training loss was almost equal (1.283 vs 1.262). Either a train/test
frame mismatch bug in the pipeline, or a real effect. Both cached example sets
come from the same ears, registrations and random draws and differ only in the
frame. Train each on the same 75% of SUBJECTS and score on the held-out 25%
(distance ||prediction - target|| is rotation invariant, so the numbers are
comparable). If aligned is worse here too, it is a real effect, not a bug.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.correction.anchor_model import train_model, predict_correction

CACHE = ROOT / "results" / "example_cache"
WORLD, LOCAL = CACHE / "examples_bbef93fb48c4089b.pkl", CACHE / "examples_d2ecc6240d7cac8f.pkl"
EPOCHS = 80


def evaluate(model, examples):
    model.eval()
    d = []
    for e in examples:
        p = predict_correction(model, e["points"], e["anchor_class"])
        d.append(np.linalg.norm(p - e["target"]))
    return np.array(d)


def main():
    w = pickle.loads(WORLD.read_bytes()); l = pickle.loads(LOCAL.read_bytes())
    assert len(w) == len(l)
    same = all(a["subject_id"] == b["subject_id"] and a["landmark_idx"] == b["landmark_idx"] and a["side"] == b["side"]
               for a, b in zip(w, l))
    tn = np.abs(np.array([np.linalg.norm(a["target"]) - np.linalg.norm(b["target"]) for a, b in zip(w, l)]))
    print(f"same example order: {same}; target length differs by max {tn.max():.2e} (rotation only)", flush=True)
    subs = sorted({e["subject_id"] for e in w})
    rng = np.random.default_rng(0); rng.shuffle(subs)
    hold = set(subs[:len(subs) // 4])
    idx_tr = [i for i, e in enumerate(w) if e["subject_id"] not in hold]
    idx_te = [i for i, e in enumerate(w) if e["subject_id"] in hold]
    print(f"{len(subs)} subjects, {len(idx_tr)} train / {len(idx_te)} held-out examples", flush=True)
    res = {}
    for seed in (0, 1):
        for name, ex in (("world", w), ("aligned", l)):
            t0 = time.time()
            tr = [ex[i] for i in idx_tr]; te = [ex[i] for i in idx_te]
            m = train_model(tr, n_epochs=EPOCHS, verbose=False, capacity="large", torch_seed=seed,
                            device=torch.device("cpu"), loss_type="wing", augment_rotate=10, augment_scale=0.1,
                            swa_start=0.75)
            dtr = evaluate(m, tr[:800]); dte = evaluate(m, te)
            res[(name, seed)] = (dtr.mean(), dte.mean(), dte)
            print(f"seed {seed} {name:8s} train {dtr.mean():.3f}  held-out {dte.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print()
    for name in ("world", "aligned"):
        print(f"{name:8s} held-out mean over seeds: {np.mean([res[(name, s)][1] for s in (0, 1)]):.3f}")
    a = np.mean([res[('aligned', s)][2] for s in (0, 1)], 0); b = np.mean([res[('world', s)][2] for s in (0, 1)], 0)
    d = a - b
    print(f"aligned - world (per held-out example): {d.mean():+.4f}  (t {d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.1f})")
    lm = np.array([e["landmark_idx"] for e in [w[i] for i in idx_te]])
    print("per anchor (aligned - world):", " ".join(f"{g}:{d[lm == g].mean():+.2f}" for g in sorted(set(lm))))


if __name__ == "__main__":
    main()
