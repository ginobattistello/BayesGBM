"""Central preflight validation for BayesGBM model specifications."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .state_model import trial_input


@dataclass(frozen=True)
class ValidatedSpec:
    n_subjects: int
    n_parameters: int
    n_state: int
    parameter_names: tuple[str, ...]
    state_names: tuple[str, ...]


def _check_numeric_finite(value, name: str):
    try:
        arr = np.asarray(value)
    except Exception:
        return
    if arr.dtype.kind in "biufc" and not np.all(np.isfinite(arr.astype(float))):
        raise ValueError(f"{name} contains NaN or Inf")


def _validate_u(u, T: int, subject: int):
    if u is None:
        return
    if isinstance(u, dict):
        for key, value in u.items():
            arr = np.asarray(value)
            if arr.ndim > 0 and arr.shape[0] != T:
                raise ValueError(
                    f"subject {subject}: input field {key!r} has length {arr.shape[0]}, expected {T}"
                )
            _check_numeric_finite(value, f"subject {subject} input {key!r}")
        return
    arr = np.asarray(u)
    if arr.ndim > 0 and arr.shape[0] != T:
        raise ValueError(f"subject {subject}: u has first dimension {arr.shape[0]}, expected {T}")
    _check_numeric_finite(u, f"subject {subject} u")


def validate_fit_spec(data, model, config) -> ValidatedSpec:
    """Normalize/check all modeller-specified information before optimization."""
    if not isinstance(data, (list, tuple)) or len(data) == 0:
        raise ValueError("data must be a non-empty list/tuple of subject dictionaries")

    # Force evaluation of combined names, which also checks cross-block uniqueness.
    parameter_names = tuple(model.priors.names)
    if len(parameter_names) != model.n_parameters:
        raise ValueError("parameter names are inconsistent with prior dimension")
    if len(model.state_names) != model.n_state:
        raise ValueError("state_names are inconsistent with initial_state")

    # Check bounds dimension if supplied.
    if config.hard_bounds is not None:
        if len(config.hard_bounds) != model.n_parameters:
            raise ValueError("hard_bounds must contain one (low, high) pair per static parameter")
        for j, pair in enumerate(config.hard_bounds):
            if pair is None:
                continue
            if len(pair) != 2:
                raise ValueError(f"hard_bounds[{j}] must be (low, high) or None")
            lo, hi = pair
            if lo is not None and hi is not None and not lo < hi:
                raise ValueError(f"hard_bounds[{j}] must satisfy low < high")

    for n, subject in enumerate(data):
        if not isinstance(subject, dict) or "y" not in subject:
            raise ValueError(f"subject {n}: each subject must be a dict containing 'y'")
        if "u" not in subject:
            raise ValueError(f"subject {n}: each subject must contain the generic input field 'u'")
        y = np.asarray(subject["y"])
        if y.ndim == 0 or y.shape[0] == 0:
            raise ValueError(f"subject {n}: y must contain at least one trial")
        _check_numeric_finite(y, f"subject {n} y")
        T = y.shape[0]
        _validate_u(subject.get("u"), T, n)

        if model.family == "bernoulli":
            yf = np.asarray(y, dtype=float).reshape(-1)
            if not np.all(np.isin(yf, [0.0, 1.0])):
                raise ValueError(f"subject {n}: Bernoulli y must contain only 0 and 1")
        elif model.family == "categorical":
            yf = np.asarray(y, dtype=float).reshape(-1)
            if not np.all(np.equal(yf, np.floor(yf))) or np.any(yf < 0):
                raise ValueError(f"subject {n}: categorical y must contain non-negative integer labels")
        else:
            try:
                np.asarray(y, dtype=float)
            except Exception as exc:
                raise ValueError(f"subject {n}: Gaussian y must be numeric") from exc

        # Representative scientific-function checks at the prior mean.
        theta = model.priors.evolution.mean.copy()
        phi = model.priors.observation.mean.copy()
        x0 = model.initial_state.copy()
        u0 = trial_input(subject.get("u"), 0, T)
        pred1 = model.observation(x0.copy(), phi.copy(), u0)
        pred2 = model.observation(x0.copy(), phi.copy(), u0)
        try:
            a, b = np.asarray(pred1, dtype=float), np.asarray(pred2, dtype=float)
        except Exception as exc:
            raise ValueError(f"subject {n}: observation must return numeric output") from exc
        if a.shape != b.shape or not np.allclose(a, b, equal_nan=True):
            raise ValueError("observation must be deterministic for fixed inputs")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"subject {n}: observation returned NaN/Inf at prior means")

        if model.family == "bernoulli":
            if a.size != 1 or float(a.reshape(-1)[0]) < 0 or float(a.reshape(-1)[0]) > 1:
                raise ValueError("Bernoulli observation must return scalar P(y=1) in [0, 1]")
        elif model.family == "categorical":
            p = a.reshape(-1)
            if p.size < 2 or np.any(p < 0) or not np.isclose(np.sum(p), 1.0, atol=1e-8):
                raise ValueError("categorical observation must return a probability vector summing to one")
            if int(np.max(np.asarray(y).reshape(-1))) >= p.size:
                raise ValueError("categorical y contains a category outside the returned probability vector")

        x1 = np.asarray(model.evolution(x0.copy(), theta.copy(), u0, y[0]), dtype=float).reshape(-1)
        x2 = np.asarray(model.evolution(x0.copy(), theta.copy(), u0, y[0]), dtype=float).reshape(-1)
        if x1.shape != x0.shape or not np.all(np.isfinite(x1)):
            raise ValueError("evolution must return a finite state with the same shape as initial_state")
        if not np.allclose(x1, x2, equal_nan=True):
            raise ValueError("evolution must be deterministic for fixed inputs")

        # Covariance shape/PSD checks at representative values.
        _ = model.initial_covariance_matrix()
        _ = model.process_covariance_matrix(theta, u0)
        if model.family == "gaussian":
            dim = a.reshape(-1).size
            _ = model.observation_covariance_matrix(phi, u0, dim)

        # Full prior-mean likelihood is the strongest preflight integration test.
        out = model.evaluate(model.priors.mean, subject)
        ll = np.asarray(out["loglik"], dtype=float)
        if ll.shape != (T,) or not np.all(np.isfinite(ll)):
            raise ValueError(f"subject {n}: prior-mean model evaluation returned invalid log likelihood")

    if config.latent_uncertainty == "filtered" and not model.has_state_uncertainty():
        raise ValueError(
            "latent_uncertainty='filtered' requires initial-state uncertainty or process_covariance; "
            "use 'propagated' or 'none' for a deterministic state model"
        )

    return ValidatedSpec(
        n_subjects=len(data),
        n_parameters=model.n_parameters,
        n_state=model.n_state,
        parameter_names=parameter_names,
        state_names=tuple(model.state_names),
    )
