# Reproduction Notes

Every result quoted in `SOLUTION_README.md`, with the exact command, config and
expected output for each. Written after a teammate could not reproduce the
interpolation table — the four most common failure modes are listed under
"If your numbers don't match" in each section.

**Read `01_CRITICAL_GOTCHAS.md` first.** Most reproduction failures come from
four specific misunderstandings, not from environment differences.

## Contents

| file | covers |
|---|---|
| `01_CRITICAL_GOTCHAS.md` | the four things that break reproduction |
| `02_ENVIRONMENT.md` | versions, hardware, determinism |
| `03_PIPELINE_CONFIG.md` | every constant in the pipeline and where it lives |
| `04_MAIN_RESULTS.md` | the 1.5931mm headline and full-CV runs |
| `05_INTERPOLATION_TABLE.md` | the blend vs prior vs prior_snap table |
| `06_DIAGNOSTICS.md` | all 15 diagnostic scripts, commands and expected output |
| `07_MEASUREMENT_PROTOCOL.md` | how to run a comparison that means something |

## Quick start

```bash
cd Example_notebook_data
pip install -r requirements.txt
python scripts/cross_validate.py --data-dir "<parent of mesh/ and landmarks/>" \
    --n-subjects 200 --outer-folds 3 --inner-folds 4 --k-references 7 \
    --epochs 150 --with-correction --with-polish \
    --capacity large --multi-seed-ensemble --n-seeds 3
```

Expect **~1.59mm** over 400 ears, in roughly 3-5 hours.

`--data-dir` is the folder CONTAINING `mesh/` and `landmarks/`, not either of
them. For the challenge data that is
`2026 Munich Tech Arena - Datas` (inside Example_notebook_data; holds mesh/ and landmarks/ directly, NOT nested).
