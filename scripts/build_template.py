"""Build the registration Template used by the default LandmarkExtractor and
save it to assets/template.npz.

Usage:
    python scripts/build_template.py --data-dir "<mesh/landmarks parent>" \
        --k-references 5 --out assets/template.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.foundations.dataset import Dataset
from src.registration.template import build_template, save_template


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="parent of mesh/ and landmarks/")
    p.add_argument("--k-references", type=int, default=5)
    p.add_argument("--n-subjects", type=int, default=None,
                    help="limit number of subjects used to build the template (default: all)")
    p.add_argument("--out", default="assets/template.npz")
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                 landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    subject_ids = ds.subject_ids if args.n_subjects is None else ds.subject_ids[:args.n_subjects]

    print(f"Building template from {len(subject_ids)} subjects, k_references={args.k_references}")
    template = build_template(ds, subject_ids, k_references=args.k_references)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_template(template, str(out_path))
    print(f"Saved template ({len(template.references)} references) to {out_path}")


if __name__ == "__main__":
    main()
