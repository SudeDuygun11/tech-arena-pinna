"""Train the Stage-1 correction network and save weights to assets/anchor_model.pt.

Training data is generated leakage-free via inner k-fold cross-fitting (see
src/training_data.py): the registration template used to score a given
subject is always built from *other* subjects, never the subject itself.

Usage:
    python scripts/train_anchor_model.py --data-dir "<mesh/landmarks parent>" \
        --inner-folds 4 --k-references 5 --epochs 40 --out assets/anchor_model.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.correction.anchor_model import train_model
from src.foundations.dataset import Dataset
from src.correction.training_data import generate_inner_cv_examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="parent of mesh/ and landmarks/")
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--k-references", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-subjects", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="assets/anchor_model.pt")
    args = p.parse_args()

    ds = Dataset(mesh_dir=str(Path(args.data_dir) / "mesh"),
                 landmarks_dir=str(Path(args.data_dir) / "landmarks"))
    subject_ids = ds.subject_ids if args.n_subjects is None else ds.subject_ids[:args.n_subjects]

    print(f"Generating leakage-free training examples from {len(subject_ids)} subjects "
          f"({args.inner_folds}-fold inner CV)...")
    examples = generate_inner_cv_examples(ds, subject_ids, k_inner=args.inner_folds,
                                           k_references=args.k_references, seed=args.seed,
                                           verbose=True)
    print(f"Training on {len(examples)} examples for {args.epochs} epochs...")
    model = train_model(examples, n_epochs=args.epochs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
