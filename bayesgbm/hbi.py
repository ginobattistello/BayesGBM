"""Hierarchical group refitting for BayesGBM.

This module provides a compact hierarchical empirical-Bayes approximation
inspired by the CBM/HBI framework of Piray et al. (2019). It repeatedly
refits subjects under model-specific Gaussian group priors, updates those
priors by responsibility-weighted posterior moment matching, and updates
model frequencies with a Dirichlet variational step.

The individual subject approximation in every refit is BayesGBM's independent
MAP/observed-Hessian Laplace approximation. The implementation is intentionally
small and transparent; it is not presented as an exact line-by-line port of
the original MATLAB/Python HBI equations.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.special import psi

from .individual_fit import individual_fit
from .optimization import Config
from .priors import GaussianPrior, Priors


@dataclass(frozen=True)
class HBIConfig:
    maxiter: int = 30
    tol: float = 1e-3
    verbose: bool = True
    random_state: int | None = 42

    def __post_init__(self):
        if self.maxiter < 1:
            raise ValueError("maxiter must be >= 1")
        if self.tol <= 0:
            raise ValueError("tol must be > 0")


@dataclass
class HBIResult:
    models: list
    fits: list
    responsibilities: np.ndarray
    dirichlet_parameters: np.ndarray
    model_frequency: np.ndarray
    group_priors: list[Priors]
    converged: bool
    n_iter: int
    method: str = "hierarchical_empirical_bayes_laplace"


def _responsibilities(lme, alpha):
    log_r = lme + (psi(alpha) - psi(np.sum(alpha)))[None, :]
    log_r -= np.max(log_r, axis=1, keepdims=True)
    r = np.exp(log_r)
    r /= np.sum(r, axis=1, keepdims=True)
    return r


def _weighted_block_prior(fit, weights, slc, original_prior: GaussianPrior):
    means = fit.output.parameters[:, slc]
    d = means.shape[1]
    if d == 0:
        return original_prior
    w = np.asarray(weights, dtype=float)
    sw = float(np.sum(w))
    if sw <= 1e-12:
        return original_prior
    mu = np.sum(w[:, None] * means, axis=0) / sw
    cov = np.zeros((d, d), dtype=float)
    for i in range(len(w)):
        full_cov = fit.math.covariance[i]
        if full_cov is None:
            raise RuntimeError("HBI requires valid subject Laplace covariance")
        Si = full_cov[slc, slc]
        delta = means[i] - mu
        cov += w[i] * (Si + np.outer(delta, delta))
    cov /= sw

    # Preserve fixed parameters from the original model exactly.
    fixed = original_prior.fixed_mask
    for j in np.flatnonzero(fixed):
        mu[j] = original_prior.mean[j]
        cov[j, :] = 0.0
        cov[:, j] = 0.0
    free = ~fixed
    if np.any(free):
        block = 0.5 * (cov[np.ix_(free, free)] + cov[np.ix_(free, free)].T)
        vals, vecs = np.linalg.eigh(block)
        floor = max(1e-8, 1e-8 * float(np.max(vals)) if vals.size else 1e-8)
        vals = np.maximum(vals, floor)
        cov[np.ix_(free, free)] = vecs @ np.diag(vals) @ vecs.T
    return GaussianPrior(mu, cov, names=original_prior.names)


def hbi_main(data, models, *, fit_config: Config | None = None, hbi_config: HBIConfig | None = None):
    """Hierarchically refit one or more StateModels across subjects."""
    if not isinstance(models, (list, tuple)) or len(models) == 0:
        raise ValueError("models must be a non-empty sequence of StateModel objects")
    if fit_config is None:
        fit_config = Config(verbose=False, display=False, latent_uncertainty="none")
    else:
        fit_config = replace(fit_config, verbose=False, display=False, latent_uncertainty="none")
    if hbi_config is None:
        hbi_config = HBIConfig()

    group_priors = [m.priors for m in models]
    alpha = np.ones(len(models), dtype=float)
    prev_freq = alpha / alpha.sum()
    prev_means = [p.mean.copy() for p in group_priors]
    converged = False
    fits = None
    r = np.full((len(data), len(models)), 1.0 / len(models))

    for it in range(1, hbi_config.maxiter + 1):
        fits = []
        lme = np.zeros((len(data), len(models)))
        current_models = []
        for k, (model, priors) in enumerate(zip(models, group_priors)):
            mk = replace(model, priors=priors)
            current_models.append(mk)
            fit = individual_fit(data, mk, config=fit_config)
            for n, diag in enumerate(fit.math.diagnostics):
                if not diag.laplace_valid:
                    raise RuntimeError(f"HBI requires valid Laplace curvature: model {k}, subject {n}")
            fits.append(fit)
            lme[:, k] = fit.output.log_evidence

        r = _responsibilities(lme, alpha)
        alpha = 1.0 + np.sum(r, axis=0)
        freq = alpha / np.sum(alpha)

        new_priors = []
        for k, (fit, original_model) in enumerate(zip(fits, models)):
            pe = _weighted_block_prior(fit, r[:, k], original_model.priors.theta_slice, original_model.priors.evolution)
            po = _weighted_block_prior(fit, r[:, k], original_model.priors.phi_slice, original_model.priors.observation)
            new_priors.append(Priors(pe, po))

        mean_change = max(float(np.linalg.norm(p.mean - old)) for p, old in zip(new_priors, prev_means))
        freq_change = float(np.linalg.norm(freq - prev_freq))
        if hbi_config.verbose:
            print(f"HBI iter {it:02d}: parameter change={mean_change:.3g}, frequency change={freq_change:.3g}")
        group_priors = new_priors
        if max(mean_change, freq_change) < hbi_config.tol:
            converged = True
            break
        prev_means = [p.mean.copy() for p in group_priors]
        prev_freq = freq.copy()

    # Final refit under final group priors so returned fits match returned priors.
    final_fits = []
    lme = np.zeros((len(data), len(models)))
    final_models = []
    for k, (model, priors) in enumerate(zip(models, group_priors)):
        mk = replace(model, priors=priors)
        final_models.append(mk)
        fit = individual_fit(data, mk, config=fit_config)
        final_fits.append(fit)
        lme[:, k] = fit.output.log_evidence
    r = _responsibilities(lme, alpha)
    alpha = 1.0 + np.sum(r, axis=0)
    freq = alpha / alpha.sum()

    return HBIResult(
        models=final_models,
        fits=final_fits,
        responsibilities=r,
        dirichlet_parameters=alpha,
        model_frequency=freq,
        group_priors=group_priors,
        converged=converged,
        n_iter=it,
    )
