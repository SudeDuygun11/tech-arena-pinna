"""Full pipeline run with the artifacts needed to reproduce it.

Recipe (confirmed on fold-0 screens, see project notes): Wing loss + rotation/scale
augmentation + SWA for the anchor networks, per-ear reference selection (best 5 of
40 by warped-surface fit), polish networks with their legacy training.

Stage 1  3-fold subject-wise cross-validation over all 200 subjects (400 ears):
           bundle_fold{0,1,2}.pt   trained anchor+polish networks, template, config,
                                   and the exact train/test subject ids of that fold
           oof_predictions.npz     out-of-fold predictions for all 400 ears
Stage 2  production model trained on ALL 200 subjects:
           bundle_final_all200.pt  (nothing held out, so no score -- the CV number
                                   in stage 1 is its estimate)
Also written to results/full_run_<tag>/:
           manifest.json           command lines, git commit, environment, seeds, fold
                                   assignment, source hashes, data fingerprint, timings
           requirements_frozen.txt pip freeze
           cv.log, final.log       full training/evaluation logs
           SHA256SUMS.txt          checksums of every artifact

Resumable: a stage whose outputs already exist is skipped. The machine is kept
awake for the duration (a per-process request, released when this script exits).

    python scripts/run_full_pipeline.py --tag recipe_v1
    python scripts/run_full_pipeline.py --tag smoke --smoke --out-root <dir>   # minutes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = "2026 Munich Tech Arena - Datas"

RECIPE = ["--loss", "wing", "--augment-rotate", "10", "--augment-scale", "0.1", "--swa-start", "0.75",
          "--k-references", "40", "--select-references", "5", "--polish-legacy"]
FULL = ["--n-subjects", "200", "--inner-folds", "4", "--epochs", "150", "--n-seeds", "3"]
SMOKE = ["--n-subjects", "9", "--inner-folds", "2", "--epochs", "3", "--n-seeds", "2"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(cmd):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception:
        return None


def keep_awake():
    if os.name == "nt":
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no system settings are changed; released at exit
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)


def data_fingerprint():
    mesh_dir, lm_dir = ROOT / DATA / "mesh", ROOT / DATA / "landmarks"
    meshes = sorted(p for p in mesh_dir.iterdir() if p.is_file()) if mesh_dir.exists() else []
    lms = sorted(p for p in lm_dir.iterdir() if p.is_file()) if lm_dir.exists() else []
    h = hashlib.sha256("\n".join(f"{p.name}:{p.stat().st_size}" for p in meshes).encode()).hexdigest()
    return {"n_meshes": len(meshes), "mesh_bytes": sum(p.stat().st_size for p in meshes),
            "mesh_name_size_sha256": h, "n_landmark_files": len(lms),
            "landmarks_content_sha256": hashlib.sha256(b"".join(sha256(p).encode() for p in lms)).hexdigest()}


def environment():
    import numpy, scipy, torch, trimesh
    return {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__, "trimesh": trimesh.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "note": "GPU training is not bit-deterministic; seeds are recorded. The registration "
                    "cache is exact (results identical with or without it)."}


def folds(n_subjects, k, seed):
    sys.path.insert(0, str(ROOT))
    from src.foundations.dataset import Dataset
    from src.foundations.splits import k_fold_subject_split
    ds = Dataset(mesh_dir=str(ROOT / DATA / "mesh"), landmarks_dir=str(ROOT / DATA / "landmarks"))
    ids = ds.subject_ids[:n_subjects]
    return [{"fold": i, "train": tr, "test": te} for i, (tr, te) in enumerate(k_fold_subject_split(ids, k, seed))]


def stage(name, cmd, log, env, manifest, mpath):
    t0 = time.time()
    print(f"[{datetime.now():%H:%M}] stage {name}: {' '.join(cmd[2:8])} ...", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    text = Path(log).read_text(encoding="utf-8", errors="ignore")
    summary = re.findall(r"\[OVERALL[^\n]*\n(?:[^\n]*mean=[^\n]*\n)*", text)
    manifest["stages"][name] = {"exit_code": rc, "minutes": round((time.time() - t0) / 60, 1),
                                "finished": datetime.now().isoformat(timespec="seconds"),
                                "overall_summary": summary[-1].strip().splitlines() if summary else None}
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"[{datetime.now():%H:%M}] stage {name}: exit {rc} after {manifest['stages'][name]['minutes']} min", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="recipe_v1")
    ap.add_argument("--out-root", default=str(ROOT / "results"))
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check of this script")
    ap.add_argument("--skip-final", action="store_true")
    args = ap.parse_args()
    keep_awake()

    out = Path(args.out_root) / f"full_run_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    size = SMOKE if args.smoke else FULL
    common = ["--data-dir", DATA, "--with-correction", "--with-polish", "--capacity", "large",
              "--multi-seed-ensemble", "--torch-seed", "0", "--seed", "0",
              "--example-cache", str((Path(args.out_root) / "example_cache").resolve() if args.smoke
                                     else ROOT / "results" / "example_cache"), *size, *RECIPE]
    if args.smoke:
        common[common.index("--k-references") + 1] = "6"
        common[common.index("--select-references") + 1] = "3"
    cv_cmd = [sys.executable, "-u", "scripts/cross_validate.py", *common, "--outer-folds", "3",
              "--save-models", str(out / "bundle.pt"), "--save-predictions", str(out / "oof_predictions.npz")]
    final_cmd = [sys.executable, "-u", "scripts/cross_validate.py", *common, "--train-fraction", "1.0",
                 "--save-models", str(out / "bundle_final_all200.pt")]
    env = dict(os.environ, REGISTRATION_CACHE_DIR=str((Path(args.out_root) / "registration_cache").resolve()
                                                       if args.smoke else ROOT / "results" / "registration_cache"))

    src_files = sorted(list((ROOT / "src").rglob("*.py")) + [ROOT / "scripts" / "cross_validate.py",
                                                            ROOT / "scripts" / "run_full_pipeline.py"])
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.update({
        "started": manifest.get("started", datetime.now().isoformat(timespec="seconds")),
        "tag": args.tag, "smoke": args.smoke,
        "recipe": "Wing + rot/scale augmentation + SWA anchors; per-ear reference selection (best 5 of 40 by "
                  "warped-surface fit); polish networks legacy (SmoothL1, no augmentation)",
        "commands": {"cv": cv_cmd, "final": final_cmd},
        "seeds": {"torch_seed": 0, "split_seed": 0, "n_seeds_per_ensemble": 3 if not args.smoke else 2},
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "git_dirty_files": run_text(["git", "status", "--porcelain"]),
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in src_files},
        "environment": environment(), "data_fingerprint": data_fingerprint(),
        "cv_folds": folds(9 if args.smoke else 200, 3, 0),
        "stages": manifest.get("stages", {}),
    })
    mpath.write_text(json.dumps(manifest, indent=2))
    (out / "requirements_frozen.txt").write_text(run_text([sys.executable, "-m", "pip", "freeze"]) or "")

    cv_done = (out / "oof_predictions.npz").exists() and all((out / f"bundle_fold{i}.pt").exists() for i in range(3))
    if cv_done:
        print("stage cv: outputs exist, skipping", flush=True)
    else:
        rc = stage("cv", cv_cmd, out / "cv.log", env, manifest, mpath)
        if rc != 0:
            sys.exit(rc)
    if not args.skip_final:
        if (out / "bundle_final_all200.pt").exists():
            print("stage final: output exists, skipping", flush=True)
        else:
            rc = stage("final", final_cmd, out / "final.log", env, manifest, mpath)
            if rc != 0:
                sys.exit(rc)

    sums = [f"{sha256(p)}  {p.name}" for p in sorted(out.iterdir())
            if p.is_file() and p.name not in ("SHA256SUMS.txt",)]
    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
