"""Individual MAP/Laplace fitting for BayesGBM StateModel objects."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .optimization import Config, OptimizationDiagnostics, OptimizationResult, optimize_map
from .validation import validate_fit_spec


@dataclass
class FitInput:
    model_name: str
    family: str
    parameter_names: tuple[str, ...]
    state_names: tuple[str, ...]
    theta_slice: slice
    phi_slice: slice
    prior_mean: np.ndarray
    prior_covariance: np.ndarray


@dataclass
class FitOutput:
    parameters: np.ndarray
    evolution_parameters: np.ndarray
    observation_parameters: np.ndarray
    log_evidence: np.ndarray
    latent: list[dict]
    prediction: list[np.ndarray]


@dataclass
class FitMath:
    log_likelihood: np.ndarray
    log_prior: np.ndarray
    log_joint: np.ndarray
    hessian: list[np.ndarray]
    covariance: list[np.ndarray | None]
    diagnostics: list[OptimizationDiagnostics]
    free_mask: list[np.ndarray]


@dataclass
class FitProfile:
    datetime: str
    config: Config


@dataclass
class FitResult:
    input: FitInput
    output: FitOutput
    math: FitMath
    profile: FitProfile
    model: Any
    data: Any
    method: str = "MAP/Laplace"

    def plot(self, subject: int = 0, **kwargs):
        from .display import plot_subject

        return plot_subject(self, subject=subject, **kwargs)

    def summary(self, subject: int = 0) -> str:
        from .reporting import fit_summary

        return fit_summary(self, subject=subject)

    def __repr__(self):
        return self.summary(0)


def _full_covariance(opt: OptimizationResult, n_params: int) -> np.ndarray | None:
    if opt.covariance is None:
        return None
    full = np.zeros((n_params, n_params), dtype=float)
    idx = np.flatnonzero(opt.free_mask)
    full[np.ix_(idx, idx)] = opt.covariance
    return full


def _latent_none(run, model):
    # Provide the latent trajectory only at the MAP despite invalid Laplace
    mean = np.asarray(run["states"], dtype=float)
    return {
        "state": {
            "mean": mean,
            "covariance": None,
            "sd": None,
            "interval_low": None,
            "interval_high": None,
            "interval_mass": None,
            "interval_method": None,
            "uncertainty_type": "none",
            "method": run.get("filtering_method", "deterministic"),
            "state_names": tuple(model.state_names),
        }
    }


def _latent_filtered(run, model):
    mean = np.asarray(run["states"], dtype=float)
    cov = np.asarray(run["state_covariance"], dtype=float)
    sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    return {
        "state": {
            "mean": mean,
            "covariance": cov,
            "sd": sd,
            "interval_low": None,
            "interval_high": None,
            "interval_mass": None,
            "interval_method": None,
            "uncertainty_type": "filtered",
            "method": run.get("filtering_method", "unknown_filter"),
            "state_names": tuple(model.state_names),
        }
    }


def _sample_static_posterior(opt: OptimizationResult, model, n_samples: int, rng):
    if not opt.diagnostics.laplace_valid or opt.covariance is None:
        raise ValueError("propagated latent uncertainty requires a valid Laplace posterior")
    free_draws = rng.multivariate_normal(opt.free_parameters, opt.covariance, size=n_samples)
    prior_mean = model.priors.mean
    draws = np.tile(prior_mean, (n_samples, 1))
    draws[:, opt.free_mask] = free_draws
    return draws


def _latent_propagated(opt, model, subject_data, config, rng):
    if not opt.diagnostics.laplace_valid:
        warnings.warn(
            "Laplace posterior is invalid; propagated latent uncertainty is unavailable. The MAP latent trajectory is retained without a shadow.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _latent_none(model.evaluate(opt.parameters, subject_data), model)
    if opt.diagnostics.laplace_fragile:
        warnings.warn("Laplace posterior is numerically fragile; propagated latent uncertainty may be sensitive.", RuntimeWarning, stacklevel=3)
    draws = _sample_static_posterior(opt, model, config.latent_samples, rng)
    trajectories = []
    for p in draws:
        run = model.evaluate(p, subject_data)
        trajectories.append(np.asarray(run["states"], dtype=float))
    samples = np.stack(trajectories, axis=0)  # S x T x n_state
    mean = np.mean(samples, axis=0)
    centered = samples - mean[None, :, :]
    cov = np.einsum("sti,stj->tij", centered, centered) / (samples.shape[0] - 1)
    sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    alpha = (1.0 - config.latent_interval) / 2.0
    low = np.quantile(samples, alpha, axis=0)
    high = np.quantile(samples, 1.0 - alpha, axis=0)
    return {
        "state": {
            "mean": mean,
            "covariance": cov,
            "sd": sd,
            "interval_low": low,
            "interval_high": high,
            "interval_mass": config.latent_interval,
            "interval_method": "empirical_quantile",
            "uncertainty_type": "propagated",
            "method": "laplace_posterior_sampling",
            "state_names": tuple(model.state_names),
        }
    }


def individual_fit(data, model, *, config: Config | None = None) -> FitResult:
    """Fit every subject under one :class:`~bayesgbm.StateModel`.

    The scientific likelihood is determined by the model. If latent-state
    uncertainty is present in the generative model, filtering is used during
    every objective evaluation regardless of ``latent_uncertainty``. The latter
    controls which latent uncertainty is retained for reporting/plotting.
    """
    if config is None:
        config = Config()
    validate_fit_spec(data, model, config)
    rng = np.random.default_rng(config.random_state)

    opts = []
    latent = []
    predictions = []
    for n, subject in enumerate(data):
        opt = optimize_map(subject, model, config, rng=rng)
        opts.append(opt)
        run_map = model.evaluate(opt.parameters, subject)
        predictions.append(np.asarray(run_map["prediction"], dtype=float))
        if config.latent_uncertainty == "propagated":
            latent_n = _latent_propagated(opt, model, subject, config, rng)
        elif config.latent_uncertainty == "filtered":
            latent_n = _latent_filtered(run_map, model)
        else:
            latent_n = _latent_none(run_map, model)
        latent.append(latent_n)
        if config.verbose:
            status = "valid" if opt.diagnostics.laplace_valid else "INVALID"
            frag = " (fragile)" if opt.diagnostics.laplace_fragile else ""
            print(f"Subject {n + 1:02d}: log joint={opt.log_joint:.3f}, Laplace={status}{frag}")

    parameters = np.vstack([o.parameters for o in opts])
    ntheta = model.n_theta
    result = FitResult(
        input=FitInput(
            model_name=model.name,
            family=model.family,
            parameter_names=tuple(model.priors.names),
            state_names=tuple(model.state_names),
            theta_slice=model.priors.theta_slice,
            phi_slice=model.priors.phi_slice,
            prior_mean=model.priors.mean.copy(),
            prior_covariance=model.priors.covariance.copy(),
        ),
        output=FitOutput(
            parameters=parameters,
            evolution_parameters=parameters[:, :ntheta],
            observation_parameters=parameters[:, ntheta:],
            log_evidence=np.asarray([o.log_evidence for o in opts], dtype=float),
            latent=latent,
            prediction=predictions,
        ),
        math=FitMath(
            log_likelihood=np.asarray([o.log_likelihood for o in opts], dtype=float),
            log_prior=np.asarray([o.log_prior for o in opts], dtype=float),
            log_joint=np.asarray([o.log_joint for o in opts], dtype=float),
            hessian=[o.hessian for o in opts],
            covariance=[_full_covariance(o, model.n_parameters) for o in opts],
            diagnostics=[o.diagnostics for o in opts],
            free_mask=[o.free_mask for o in opts],
        ),
        profile=FitProfile(datetime=datetime.now().isoformat(timespec="seconds"), config=config),
        model=model,
        data=copy.deepcopy(data),
    )

    if config.display:
        result.plot(subject=0, display=True)
    return result
