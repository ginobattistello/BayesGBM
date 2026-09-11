"""Generative state-model definition and deterministic likelihood helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
import numpy as np

from .priors import Priors, covariance_matrix


Family = str


def trial_input(u: Any, t: int, T: int):
    """Extract modeller-defined inputs for trial ``t`` without reserved names."""
    if u is None:
        return None
    if isinstance(u, dict):
        out = {}
        for key, value in u.items():
            arr = np.asarray(value) if not isinstance(value, str) else np.asarray(value, dtype=object)
            if arr.ndim == 0:
                out[key] = value
            else:
                if arr.shape[0] != T:
                    raise ValueError(
                        f"input field {key!r} has first dimension {arr.shape[0]}, expected {T}"
                    )
                out[key] = value[t]
        return out
    arr = np.asarray(u)
    if arr.ndim == 0:
        return u
    if arr.shape[0] != T:
        raise ValueError(f"trial input has first dimension {arr.shape[0]}, expected {T}")
    return u[t]


def _bernoulli_loglik(y, p) -> float:
    p = float(p)
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError("Bernoulli observation must return a probability in [0, 1]")
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1.0 - eps)
    return float(y * np.log(p) + (1 - y) * np.log1p(-p))


def _categorical_loglik(y, p) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size < 2 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("categorical observation must return finite non-negative probabilities")
    s = float(np.sum(p))
    if not np.isclose(s, 1.0, atol=1e-8, rtol=1e-7):
        raise ValueError("categorical probabilities must sum to one")
    y = int(y)
    if y < 0 or y >= p.size:
        raise ValueError("categorical outcome is outside the observation probability vector")
    return float(np.log(np.clip(p[y], np.finfo(float).tiny, 1.0)))


def _gaussian_loglik(y, mean, cov) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    if y.shape != mean.shape:
        raise ValueError("Gaussian observation mean has incompatible dimension")
    cov = covariance_matrix(cov, y.size, name="observation covariance", allow_semidefinite=False)
    delta = y - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("observation covariance must be positive definite")
    return float(-0.5 * (y.size * np.log(2 * np.pi) + logdet + delta @ np.linalg.solve(cov, delta)))


@dataclass(frozen=True)
class StateModel:
    """Generative model used by all BayesGBM inference routines.

    Trial order is ``x_t -> observation -> likelihood(y_t) -> evolution -> x_{t+1}``.
    """

    evolution: Callable
    observation: Callable
    family: Family
    priors: Priors
    initial_state: Sequence[float]
    initial_state_covariance: Optional[object] = None
    process_covariance: Optional[object] = None
    observation_covariance: Optional[object] = None
    state_names: Optional[Sequence[str]] = None
    name: Optional[str] = None

    def __post_init__(self):
        family = str(self.family).lower()
        if family == "binomial":
            family = "bernoulli"
        if family not in {"gaussian", "bernoulli", "categorical"}:
            raise ValueError("family must be 'gaussian', 'bernoulli', or 'categorical'")
        x0 = np.asarray(self.initial_state, dtype=float).reshape(-1)
        if x0.size == 0 or not np.all(np.isfinite(x0)):
            raise ValueError("initial_state must be a non-empty finite vector")
        if self.state_names is None:
            state_names = tuple(f"x[{i}]" for i in range(x0.size))
        else:
            state_names = tuple(str(x) for x in self.state_names)
            if len(state_names) != x0.size:
                raise ValueError("state_names must contain one name per latent-state dimension")
            if len(set(state_names)) != len(state_names):
                raise ValueError("state_names must be unique")
        if family != "gaussian" and self.observation_covariance is not None:
            raise ValueError("observation_covariance is only used for Gaussian outcomes")
        if family == "gaussian" and self.observation_covariance is None:
            raise ValueError("Gaussian models require observation_covariance")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "initial_state", x0)
        object.__setattr__(self, "state_names", state_names)
        object.__setattr__(self, "name", self.name or "StateModel")

    @property
    def n_state(self) -> int:
        return int(self.initial_state.size)

    @property
    def n_theta(self) -> int:
        return self.priors.evolution.dim

    @property
    def n_phi(self) -> int:
        return self.priors.observation.dim

    @property
    def n_parameters(self) -> int:
        return self.priors.dim

    def split_parameters(self, parameters):
        p = np.asarray(parameters, dtype=float).reshape(-1)
        if p.shape != (self.n_parameters,):
            raise ValueError(f"expected {self.n_parameters} static parameters, got {p.size}")
        return p[self.priors.theta_slice], p[self.priors.phi_slice]

    def initial_covariance_matrix(self) -> np.ndarray:
        if self.initial_state_covariance is None:
            return np.zeros((self.n_state, self.n_state), dtype=float)
        return covariance_matrix(
            self.initial_state_covariance,
            self.n_state,
            name="initial_state_covariance",
            allow_semidefinite=True,
        )

    def process_covariance_matrix(self, theta, u_t) -> np.ndarray:
        value = self.process_covariance(theta, u_t) if callable(self.process_covariance) else self.process_covariance
        if value is None:
            return np.zeros((self.n_state, self.n_state), dtype=float)
        return covariance_matrix(value, self.n_state, name="process_covariance", allow_semidefinite=True)

    def observation_covariance_matrix(self, phi, u_t, dim: int) -> np.ndarray:
        value = self.observation_covariance(phi, u_t) if callable(self.observation_covariance) else self.observation_covariance
        return covariance_matrix(value, dim, name="observation_covariance", allow_semidefinite=False)

    def has_state_uncertainty(self) -> bool:
        if np.any(self.initial_covariance_matrix() != 0.0):
            return True
        if self.process_covariance is None:
            return False
        if callable(self.process_covariance):
            return True
        return bool(np.any(np.asarray(self.process_covariance, dtype=float) != 0.0))

    def deterministic_run(self, parameters, subject_data):
        """Run a deterministic state recursion and return likelihood and traces."""
        theta, phi = self.split_parameters(parameters)
        y = np.asarray(subject_data["y"])
        u = subject_data.get("u", None)
        T = len(y)
        x = self.initial_state.copy()
        states = np.zeros((T, self.n_state), dtype=float)
        predictions = []
        loglik = np.zeros(T, dtype=float)
        for t in range(T):
            u_t = trial_input(u, t, T)
            states[t] = x
            pred = self.observation(x.copy(), phi.copy(), u_t)
            if self.family == "bernoulli":
                pred_value = float(np.asarray(pred).reshape(()))
                loglik[t] = _bernoulli_loglik(int(y[t]), pred_value)
                predictions.append(pred_value)
            elif self.family == "categorical":
                pred_value = np.asarray(pred, dtype=float).reshape(-1)
                loglik[t] = _categorical_loglik(int(y[t]), pred_value)
                predictions.append(pred_value)
            else:
                pred_value = np.asarray(pred, dtype=float).reshape(-1)
                y_t = np.asarray(y[t], dtype=float).reshape(-1)
                R = self.observation_covariance_matrix(phi, u_t, pred_value.size)
                loglik[t] = _gaussian_loglik(y_t, pred_value, R)
                predictions.append(pred_value)
            x_next = np.asarray(self.evolution(x.copy(), theta.copy(), u_t, y[t]), dtype=float).reshape(-1)
            if x_next.shape != x.shape or not np.all(np.isfinite(x_next)):
                raise ValueError("evolution must return a finite state with the same shape as x")
            x = x_next
        pred_arr = np.asarray(predictions, dtype=float)
        if self.family == "gaussian" and pred_arr.shape[1] == 1:
            pred_arr = pred_arr[:, 0]
        return {
            "loglik": loglik,
            "states": states,
            "state_covariance": np.zeros((T, self.n_state, self.n_state)),
            "prediction": pred_arr,
            "filtering_method": "deterministic",
        }

    def evaluate(self, parameters, subject_data):
        """Evaluate the generative likelihood using recursion or filtering."""
        if self.has_state_uncertainty():
            from .filtering import nonlinear_state_filter
            return nonlinear_state_filter(self, parameters, subject_data)
        return self.deterministic_run(parameters, subject_data)

    def __call__(self, parameters, subject_data):
        return self.evaluate(parameters, subject_data)["loglik"]
