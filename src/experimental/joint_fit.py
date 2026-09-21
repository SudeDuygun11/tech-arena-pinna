"""Global shape-prior constraint over the 15 anchors.

Today the anchors are corrected INDEPENDENTLY -- nothing enforces that the 15
together form a plausible ear. This module adds that constraint as a
post-processing step between anchor correction and blend interpolation.

All parameters here were fixed by measurement, not tuning-by-hand:

  D1  PCA/PDM on all 85 points, 45 modes. Reconstruction of held-out TRUE
      configurations: 0.377mm at 45/255 modes, vs 0.679mm for a 15-anchor
      model at 20/45 modes. The 85-point model is both sharper and a stronger
      constraint -- the 70 interpolated points act as extra observations that
      regularise the fit. Mode count by cross-validation, not a variance
      threshold (arXiv:1808.00309).

  D2  Per-anchor confidence from E-CPV (ensemble coordinate prediction
      variance) -- the strong baseline of arXiv:2203.02351, whose other two
      measures are heatmap-derived and inapplicable since we regress
      coordinates. Spearman(E-CPV, error) = +0.310, monotonic quintiles,
      positive on all four contours.

  D3  Prior strength 0.05. Measured sweep at STRONG local accuracy
      (1.759mm): 0.05 -> 1.714mm with confidence weighting. Note the earlier
      sweep at WEAK local accuracy (3.080mm) put the optimum at 2.0 with a
      -12.7% gain; both the optimum and the gain shrink as the local model
      improves, so 0.05 belongs with a good local model.

CRITICAL: with UNIFORM weights the prior never helps at strong local accuracy
-- every strength tested was worse than no prior at all (1.774mm vs 1.759mm at
strength 0.05). It only works with E-CPV weighting. An ensemble is therefore
required; with a single model there is no E-CPV and the prior must be skipped.


ROLE IN THE PIPELINE
--------------------
Reading order : 17 of 20   (experimental)
Duty          : PCA shape prior over the 15 anchors. MEASURED NULL.

Weighted MAP fit to a 45-mode point distribution model, weights from
ensemble prediction variance.

Known issues / status:
  -0.0072 +- 0.0164 at 400 ears: not significant. Kept for the documented
  reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..foundations.contours import ANCHOR_INDICES
from ..foundations.geometry import apply_rigid, kabsch

DEFAULT_N_MODES = 45
DEFAULT_STRENGTH = 0.05
PROCRUSTES_ITERS = 5


@dataclass
class ShapePrior:
    mu: np.ndarray            # (255,) mean of aligned 85-point shapes
    P: np.ndarray             # (k, 255) principal directions
    evals: np.ndarray         # (k,) eigenvalues
    mean_shape: np.ndarray    # (85, 3) alignment target
    calib: tuple              # (a, b): err ~ a * ecpv + b


def _procrustes_align(shapes: np.ndarray, n_iter: int = PROCRUSTES_ITERS):
    """Rigid-only generalised Procrustes. Rotation and translation but NOT
    scale: ear size (66-82mm span) is real anatomical signal and the metric is
    in millimetres, so normalising scale away would discard information."""
    aligned = shapes.copy()
    mean = aligned[0]
    for _ in range(n_iter):
        for i in range(len(aligned)):
            R, t = kabsch(aligned[i], mean)
            aligned[i] = apply_rigid(aligned[i], R, t)
        mean = aligned.mean(axis=0)
    return aligned, mean


def fit_shape_prior(train_landmarks: np.ndarray, n_modes: int = DEFAULT_N_MODES,
                     calib: tuple = (1.0, 0.0)) -> ShapePrior:
    """train_landmarks: (N, 85, 3) ground-truth configurations from TRAINING
    subjects only. `calib` maps E-CPV to an error magnitude and should come
    from calibrate_confidence() on leakage-free predictions."""
    aligned, mean_shape = _procrustes_align(np.asarray(train_landmarks, dtype=float).copy())
    X = aligned.reshape(len(aligned), -1)
    mu = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    k = int(min(n_modes, len(Vt)))
    return ShapePrior(mu=mu, P=Vt[:k], evals=(S[:k] ** 2) / max(len(X) - 1, 1),
                       mean_shape=mean_shape, calib=calib)


def calibrate_confidence(ecpv: np.ndarray, errors: np.ndarray) -> tuple:
    """Least-squares fit of err ~ a * ecpv + b.

    E-CPV correlates with error but is not calibrated to it, and
    inverse-variance weighting needs an actual magnitude. Fit this on
    leakage-free TRAINING predictions -- never on the ears being scored.

    The intercept is diagnostic: it is the error E-CPV cannot see. It fell
    from 2.108mm (weak local model) to 0.361mm (strong one), which is why
    confidence weighting matters more as the model improves.
    """
    ecpv = np.asarray(ecpv, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if len(ecpv) < 8 or np.std(ecpv) < 1e-9:
        return (1.0, 0.0)
    a, b = np.polyfit(ecpv, errors, 1)
    return (float(a), float(b))


def apply_shape_prior(corrected_anchors: dict, ecpv: dict | None, prior: ShapePrior,
                       strength: float = DEFAULT_STRENGTH) -> dict:
    """Pull the 15 corrected anchors toward a plausible ear configuration.

    Solves the weighted MAP fit
        b = (P_o^T W P_o + strength * Lambda^-1)^-1 P_o^T W (obs - mu_o)
    over the observed anchor rows, then reads the constrained anchors back out
    of the reconstructed 85-point shape.

    ecpv: {anchor_idx: spread} for inverse-variance weighting. If None, the
    fit falls back to uniform weights -- which was measured to be WORSE than
    applying no prior at all, so callers should skip the prior entirely rather
    than pass None. Kept only so the failure is explicit rather than silent.
    """
    idx = list(ANCHOR_INDICES)
    obs = np.array([corrected_anchors[a] for a in idx], dtype=float)

    # into the prior's frame
    R, t = kabsch(obs, prior.mean_shape[idx])
    obs_a = apply_rigid(obs, R, t)

    rows = np.concatenate([[3 * a, 3 * a + 1, 3 * a + 2] for a in idx])
    Po = prior.P[:, rows]
    resid = obs_a.reshape(-1) - prior.mu[rows]

    if ecpv is None:
        w = np.ones(len(rows))
    else:
        a_c, b_c = prior.calib
        sigma = np.maximum(a_c * np.array([ecpv[a] for a in idx]) + b_c, 1e-3)
        w = np.repeat(1.0 / sigma ** 2, 3)
        w = w / w.mean()

    A = (Po * w[None, :]) @ Po.T + strength * np.diag(1.0 / np.maximum(prior.evals, 1e-9))
    coef = np.linalg.solve(A, (Po * w[None, :]) @ resid)
    fitted = (prior.mu + prior.P.T @ coef).reshape(-1, 3)[idx]

    # back to the mesh frame. kabsch gives y = x @ R.T + t, so the inverse is
    # x = (y - t) @ R, i.e. apply_rigid(y, R.T, -t @ R).
    out = apply_rigid(fitted, R.T, -t @ R)
    return {a: out[i] for i, a in enumerate(idx)}
