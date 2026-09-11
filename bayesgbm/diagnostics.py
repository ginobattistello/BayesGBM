"""Numerical diagnostics for BayesGBM individual fits."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ConvergenceSummary:
    n_starts: int
    n_successful: int
    n_objective_agreeing: int
    agreement_fraction: float
    log_joint_spread: float
    max_parameter_distance: float
    same_solution: bool
    winning_success: bool
    winning_status: int | None
    winning_message: str
    gradient_norm: float


@dataclass
class HessianDiagnostics:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    posterior_sd: np.ndarray | None
    posterior_correlation: np.ndarray | None
    laplace_valid: bool
    laplace_fragile: bool


@dataclass
class DataInformationSpectrum:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    parameter_names: tuple[str, ...]
    negative_curvature: bool
    data_informed: np.ndarray


@dataclass
class LocalIdentifiability:
    singular_values: np.ndarray
    rank: int
    nullity: int
    condition_number: float
    parameter_directions: np.ndarray
    parameter_names: tuple[str, ...]


def convergence_diagnostics(fit, subject: int = 0, *, objective_tol=1e-4, parameter_tol=1e-3) -> ConvergenceSummary:
    diag = fit.math.diagnostics[subject]
    starts = list(diag.starts)
    if not starts:
        return ConvergenceSummary(
            n_starts=0, n_successful=1, n_objective_agreeing=1, agreement_fraction=1.0,
            log_joint_spread=0.0, max_parameter_distance=0.0, same_solution=True,
            winning_success=True, winning_status=diag.lbfgsb_status,
            winning_message=diag.lbfgsb_message, gradient_norm=diag.abs_grad,
        )
    vals = np.asarray([s.log_joint for s in starts], dtype=float)
    finite = np.isfinite(vals)
    best = np.max(vals[finite])
    agree = finite & (np.abs(vals - best) <= objective_tol * (1.0 + abs(best)))
    endpoints = np.asarray([s.final_parameters for s in starts], dtype=float)
    maxdist = 0.0
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            maxdist = max(maxdist, float(np.linalg.norm(endpoints[i] - endpoints[j])))
    spread = float(np.max(vals[finite]) - np.min(vals[finite])) if np.any(finite) else np.inf
    same = bool(np.all(agree) and maxdist <= parameter_tol * (1.0 + np.linalg.norm(fit.output.parameters[subject])))
    return ConvergenceSummary(
        n_starts=len(starts),
        n_successful=sum(s.success for s in starts),
        n_objective_agreeing=int(np.sum(agree)),
        agreement_fraction=float(np.mean(agree)),
        log_joint_spread=spread,
        max_parameter_distance=maxdist,
        same_solution=same,
        winning_success=diag.lbfgsb_success,
        winning_status=diag.lbfgsb_status,
        winning_message=diag.lbfgsb_message,
        gradient_norm=diag.abs_grad,
    )


def posterior_hessian_diagnostics(fit, subject: int = 0) -> HessianDiagnostics:
    H = np.asarray(fit.math.hessian[subject], dtype=float)
    if H.size == 0:
        return HessianDiagnostics(
            eigenvalues=np.zeros(0), eigenvectors=np.zeros((0, 0)), condition_number=1.0,
            posterior_sd=np.zeros(0), posterior_correlation=np.zeros((0, 0)),
            laplace_valid=True, laplace_fragile=False,
        )
    vals, vecs = np.linalg.eigh(0.5 * (H + H.T))
    diag = fit.math.diagnostics[subject]
    cov_full = fit.math.covariance[subject]
    free = fit.math.free_mask[subject]
    if cov_full is None:
        sd = corr = None
    else:
        cov = cov_full[np.ix_(free, free)]
        sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
        denom = np.outer(sd, sd)
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
        np.fill_diagonal(corr, 1.0)
    return HessianDiagnostics(
        eigenvalues=vals,
        eigenvectors=vecs,
        condition_number=diag.hess_condition_number,
        posterior_sd=sd,
        posterior_correlation=corr,
        laplace_valid=diag.laplace_valid,
        laplace_fragile=diag.laplace_fragile,
    )


def prior_preconditioned_information(fit, subject: int = 0) -> DataInformationSpectrum:
    free = fit.math.free_mask[subject]
    Hpost = np.asarray(fit.math.hessian[subject], dtype=float)
    Sigma0 = fit.input.prior_covariance[np.ix_(free, free)]
    if Hpost.size == 0:
        return DataInformationSpectrum(np.zeros(0), np.zeros((0, 0)), tuple(), False, np.zeros(0, dtype=bool))
    P0 = np.linalg.inv(Sigma0)
    Hlik = 0.5 * ((Hpost - P0) + (Hpost - P0).T)
    vals0, vecs0 = np.linalg.eigh(Sigma0)
    sqrtS = vecs0 @ np.diag(np.sqrt(vals0)) @ vecs0.T
    Htilde = 0.5 * (sqrtS @ Hlik @ sqrtS + (sqrtS @ Hlik @ sqrtS).T)
    vals, vecs = np.linalg.eigh(Htilde)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    names = tuple(np.asarray(fit.input.parameter_names, dtype=object)[free])
    return DataInformationSpectrum(
        eigenvalues=vals,
        eigenvectors=vecs,
        parameter_names=names,
        negative_curvature=bool(np.any(vals < 0.0)),
        data_informed=vals > 1.0,
    )


def _prediction_vector(model, parameters, subject_data):
    p = np.asarray(model.evaluate(parameters, subject_data)["prediction"], dtype=float)
    if model.family == "categorical":
        if p.ndim != 2 or p.shape[1] < 2:
            raise ValueError("categorical predictions must be T x K")
        p = p[:, :-1]  # remove simplex redundancy
    return p.reshape(-1)


def numerical_local_identifiability(fit, subject: int = 0, *, relative_step=1e-5, rank_tol=None) -> LocalIdentifiability:
    p0 = np.asarray(fit.output.parameters[subject], dtype=float)
    free = fit.math.free_mask[subject]
    idx = np.flatnonzero(free)
    base = _prediction_vector(fit.model, p0, fit.data[subject])
    J = np.zeros((base.size, len(idx)), dtype=float)
    for col, j in enumerate(idx):
        h = relative_step * max(1.0, abs(float(p0[j])))
        pp = p0.copy(); pp[j] += h
        pm = p0.copy(); pm[j] -= h
        J[:, col] = (
            _prediction_vector(fit.model, pp, fit.data[subject])
            - _prediction_vector(fit.model, pm, fit.data[subject])
        ) / (2.0 * h)
    if J.size == 0:
        s = np.zeros(0); Vt = np.zeros((0, 0)); rank = 0
    else:
        _, s, Vt = np.linalg.svd(J, full_matrices=False)
        tol = rank_tol if rank_tol is not None else max(J.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
        rank = int(np.sum(s > tol))
    cond = np.inf if s.size == 0 or rank < len(idx) or s[-1] == 0 else float(s[0] / s[-1])
    names = tuple(np.asarray(fit.input.parameter_names, dtype=object)[free])
    return LocalIdentifiability(
        singular_values=s,
        rank=rank,
        nullity=len(idx) - rank,
        condition_number=cond,
        parameter_directions=Vt,
        parameter_names=names,
    )
