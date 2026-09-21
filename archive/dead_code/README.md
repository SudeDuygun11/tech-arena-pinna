# Dead code, moved out of src/ on 2026-08-29

Both were pure skeletons: every function raised `NotImplementedError`, and
nothing in the project imported either of them.

| file | lines | why it was removed |
|---|---|---|
| `candidates.py` | 150 | 8 `NotImplementedError`, 0 importers. Candidate-generation scaffolding for the ranking approach, which measured a 0.534mm oracle against a 5.524mm actual selector and was abandoned. |
| `surface_query.py` | 152 | 5 `NotImplementedError`, 0 importers. Skeleton for a surface-query transformer; superseded by `src/multiview/`, which addresses the same selection problem and reached 1.510mm on landmark 0. |

Safe to delete this whole folder whenever you like. Kept only so the removal
is reversible.
