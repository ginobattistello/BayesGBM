"""Prior and posterior predictive simulation using the fitted StateModel."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from .state_model import trial_input


@dataclass
class PredictiveResult:
    parameters: np.ndarray
    replicated_data: list[dict]


def _draw_gaussian(mean, cov, rng):
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    if mean.size == 0:
        return mean.copy()
    if np.allclose(cov, 0.0):
        return mean.copy()
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    vals = np.maximum(vals, 0.0)
    return mean + vecs @ (np.sqrt(vals) * rng.normal(size=mean.size))


def simulate_subject(model, parameters, template_data, *, rng):
    """Generate one replicated subject using the model's own generative equations."""
    theta, phi = model.split_parameters(parameters)
    y_template = np.asarray(template_data["y"])
    T = y_template.shape[0]
    u = copy.deepcopy(template_data.get("u", None))
    x = _draw_gaussian(model.initial_state, model.initial_covariance_matrix(), rng)
    ys = []
    states = np.zeros((T, model.n_state))
    for t in range(T):
        u_t = trial_input(u, t, T)
        states[t] = x
        pred = model.observation(x.copy(), phi.copy(), u_t)
        if model.family == "bernoulli":
            p = float(np.asarray(pred).reshape(()))
            y_t = int(rng.random() < p)
        elif model.family == "categorical":
            p = np.asarray(pred, dtype=float).reshape(-1)
            p = p / p.sum()
            y_t = int(rng.choice(len(p), p=p))
        else:
            mu = np.asarray(pred, dtype=float).reshape(-1)
            R = model.observation_covariance_matrix(phi, u_t, mu.size)
            draw = rng.multivariate_normal(mu, R)
            y_t = float(draw[0]) if mu.size == 1 else draw
        ys.append(y_t)
        x_det = np.asarray(model.evolution(x.copy(), theta.copy(), u_t, y_t), dtype=float).reshape(-1)
        Q = model.process_covariance_matrix(theta, u_t)
        x = _draw_gaussian(x_det, Q, rng)
    return {"y": np.asarray(ys), "u": u, "latent": states}


def prior_predictive(model, template_data, *, n_samples=100, random_state=42) -> PredictiveResult:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    rng = np.random.default_rng(random_state)
    mean = model.priors.mean
    cov = model.priors.covariance
    params = np.stack([_draw_gaussian(mean, cov, rng) for _ in range(n_samples)])
    reps = [simulate_subject(model, p, template_data, rng=rng) for p in params]
    return PredictiveResult(parameters=params, replicated_data=reps)


def posterior_predictive(fit, subject=0, *, n_samples=100, random_state=42) -> PredictiveResult:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    diag = fit.math.diagnostics[subject]
    cov = fit.math.covariance[subject]
    if not diag.laplace_valid or cov is None:
        raise ValueError("posterior predictive checks require a valid Laplace posterior")
    rng = np.random.default_rng(random_state)
    mean = fit.output.parameters[subject]
    # Fixed dimensions have zero posterior covariance and remain fixed automatically.
    params = np.stack([_draw_gaussian(mean, cov, rng) for _ in range(n_samples)])
    reps = [simulate_subject(fit.model, p, fit.data[subject], rng=rng) for p in params]
    return PredictiveResult(parameters=params, replicated_data=reps)
