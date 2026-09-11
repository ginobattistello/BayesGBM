"""MAP optimization and independent observed-Hessian Laplace inference."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Individual-fit configuration.

    ``latent_samples`` and ``latent_interval`` are retained only when
    ``latent_uncertainty='propagated'``; for all other modes they are
    resolved to ``None`` so they cannot affect computation or metadata.
    """

    num_init: int = 5
    random_state: int | None = 42
    verbose: bool = True
    display: bool = False
    maxiter: int = 1000

    hard_bounds: Sequence[tuple[float | None, float | None] | None] | None = None

    hessian_method: str = "central_fd"
    hessian_relative_step: float = 1e-4
    condition_number_warn: float = 1e12

    latent_uncertainty: str = "none"
    latent_samples: int | None = 1000
    latent_interval: float | None = 0.95

    def __post_init__(self):
        if not isinstance(self.num_init, int) or self.num_init < 1:
            raise ValueError("num_init must be a positive integer")
        if not isinstance(self.maxiter, int) or self.maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        if self.random_state is not None and not isinstance(self.random_state, (int, np.integer)):
            raise ValueError("random_state must be an integer or None")
        if self.hessian_method != "central_fd":
            raise ValueError("the current BayesGBM release supports hessian_method='central_fd'")
        if self.hessian_relative_step <= 0:
            raise ValueError("hessian_relative_step must be > 0")
        if self.condition_number_warn <= 1:
            raise ValueError("condition_number_warn must be > 1")

        mode = str(self.latent_uncertainty).lower()
        if mode not in {"none", "propagated", "filtered"}:
            raise ValueError("latent_uncertainty must be 'none', 'propagated', or 'filtered'")
        object.__setattr__(self, "latent_uncertainty", mode)

        if mode == "propagated":
            if self.latent_samples is None or int(self.latent_samples) < 2:
                raise ValueError("latent_samples must be >= 2 for propagated uncertainty")
            if self.latent_interval is None or not (0 < float(self.latent_interval) < 1):
                raise ValueError("latent_interval must lie in (0, 1) for propagated uncertainty")
            object.__setattr__(self, "latent_samples", int(self.latent_samples))
            object.__setattr__(self, "latent_interval", float(self.latent_interval))

        else:
            object.__setattr__(self, "latent_samples", None)
            object.__setattr__(self, "latent_interval", None)


# ---------------------------------------------------------------------
# Optimization result containers
# ---------------------------------------------------------------------
@dataclass
class StartRecord:
    """Summary of one L-BFGS-B initialization."""

    initial_parameters: np.ndarray
    final_parameters: np.ndarray
    log_joint: float
    success: bool
    status: int
    message: str
    n_iter: int
    gradient_norm: float


@dataclass
class OptimizationDiagnostics:
    """Optimization and Laplace diagnostics."""

    starts: list[StartRecord] = field(default_factory=list)

    search_path: np.ndarray | None = None
    search_log_joint: np.ndarray | None = None

    lbfgsb_success: bool = False
    lbfgsb_status: int | None = None
    lbfgsb_message: str = ""

    abs_grad: float = np.nan

    at_hard_bounds: np.ndarray | None = None

    hess_method: str = "central_fd"
    hess_raw_min_eig: float = np.nan
    hess_raw_max_eig: float = np.nan
    hess_condition_number: float = np.nan
    hess_ill_conditioned: bool = False

    laplace_valid: bool = False
    laplace_fragile: bool = False


@dataclass
class OptimizationResult:
    """Result of one subject-level MAP fit."""

    parameters: np.ndarray
    free_parameters: np.ndarray

    log_likelihood: float
    log_prior: float
    log_joint: float

    hessian: np.ndarray
    covariance: np.ndarray | None

    log_evidence: float

    diagnostics: OptimizationDiagnostics
    free_mask: np.ndarray


# ---------------------------------------------------------------------
# Free-parameter representation
# ---------------------------------------------------------------------
def _free_space(priors):
    """Construct the reduced free-parameter representation.

    Static parameters with zero prior variance are treated as fixed at
    their prior mean and removed from the optimization space.
    """
    mean = np.asarray(priors.mean, dtype=float)
    covariance = np.asarray(priors.covariance, dtype=float)
    fixed_mask = np.isclose(np.diag(covariance), 0.0, atol=1e-14, rtol=0.0)
    free_mask = ~fixed_mask
    free_mean = mean[free_mask]
    free_covariance = covariance[np.ix_(free_mask, free_mask)]
    if free_covariance.size:
        eig = np.linalg.eigvalsh(free_covariance)
        if np.min(eig) <= 0:
            raise ValueError("free-parameter prior covariance must be positive definite")
        precision = np.linalg.inv(free_covariance)
        sign, logdet = np.linalg.slogdet(free_covariance)
        if sign <= 0:
            raise ValueError("free-parameter prior covariance must be positive definite")
        logdet_covariance = float(logdet)
    else:
        precision = np.zeros((0, 0), dtype=float)
        logdet_covariance = 0.0
    return (mean, free_mask, free_mean, free_covariance, precision, logdet_covariance)


def _reconstruct(free_values, mean, free_mask):
    """Reconstruct the full static parameter vector."""
    full = mean.copy()
    full[free_mask] = np.asarray(free_values, dtype=float)
    return full


# ---------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------
def _log_prior_free(x, mean, precision, logdet_covariance):
    """Gaussian log prior in free-parameter space."""
    d = len(mean)
    if d == 0:
        return 0.0
    delta = np.asarray(x, dtype=float) - mean
    quadratic = delta @ precision @ delta
    return float(-0.5 * (d * np.log(2.0 * np.pi) + logdet_covariance + quadratic))


# ---------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------
def _free_bounds(config: Config, free_mask):
    """Project full-space hard bounds into free-parameter space."""
    if config.hard_bounds is None:
        return None
    out = []
    for is_free, pair in zip(free_mask, config.hard_bounds):
        if not is_free:
            continue
        if pair is None:
            out.append((None, None))
        else:
            out.append(tuple(pair))
    return out


def _clip_to_bounds(x, bounds):
    """Move an initial point inside the L-BFGS-B bounds."""
    x = np.asarray(x, dtype=float).copy()
    if bounds is None:
        return x
    for j, (lo, hi) in enumerate(bounds):
        if lo is not None:
            x[j] = max(x[j], lo + 1e-10)
        if hi is not None:
            x[j] = min(x[j], hi - 1e-10)
    return x


def _at_hard_bounds(parameters, hard_bounds):
    """Identify fitted parameters lying on explicit hard bounds."""
    at_bounds = np.zeros(len(parameters), dtype=bool)
    if hard_bounds is None:
        return at_bounds
    for j, (value, pair) in enumerate(zip(parameters, hard_bounds)):
        if pair is None:
            continue
        lo, hi = pair
        tol = 1e-7 * max(1.0, abs(float(value)))
        if lo is not None and abs(value - lo) <= tol:
            at_bounds[j] = True
        if hi is not None and abs(value - hi) <= tol:
            at_bounds[j] = True
    return at_bounds


# ---------------------------------------------------------------------
# Numerical Hessian
# ---------------------------------------------------------------------
def central_hessian(func, x, relative_step=1e-4):
    """Central finite-difference Hessian of a scalar function."""

    x = np.asarray(x, dtype=float).reshape(-1)
    d = x.size
    H = np.zeros((d, d), dtype=float)
    if d == 0:
        return H
    h = relative_step * np.maximum(1.0, np.abs(x))
    f0 = float(func(x))
    if not np.isfinite(f0):
        raise FloatingPointError("objective is non-finite at MAP during Hessian calculation")

    # Diagonal terms
    for i in range(d):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h[i]
        xm[i] -= h[i]
        fp = float(func(xp))
        fm = float(func(xm))
        if not np.isfinite(fp) or not np.isfinite(fm):
            raise FloatingPointError("objective is non-finite near MAP during Hessian calculation")
        H[i, i] = (fp - 2.0 * f0 + fm) / (h[i] ** 2)

        # Off-diagonal terms
        for j in range(i + 1, d):
            xpp = x.copy()
            xpm = x.copy()
            xmp = x.copy()
            xmm = x.copy()
            xpp[i] += h[i]
            xpp[j] += h[j]
            xpm[i] += h[i]
            xpm[j] -= h[j]
            xmp[i] -= h[i]
            xmp[j] += h[j]
            xmm[i] -= h[i]
            xmm[j] -= h[j]

            vals = np.array([float(func(xpp)), float(func(xpm)), float(func(xmp)), float(func(xmm))], dtype=float)

            if not np.all(np.isfinite(vals)):
                raise FloatingPointError("objective is non-finite near MAP during Hessian calculation")

            Hij = (vals[0] - vals[1] - vals[2] + vals[3]) / (4.0 * h[i] * h[j])

            H[i, j] = Hij
            H[j, i] = Hij

    # Explicit symmetrization protects against tiny numerical asymmetries.
    return 0.5 * (H + H.T)


# ---------------------------------------------------------------------
# MAP optimization
# ---------------------------------------------------------------------
def optimize_map(subject_data, model, config: Config, *, rng=None) -> OptimizationResult:
    """Fit one subject using multi-start L-BFGS-B."""

    if rng is None:
        rng = np.random.default_rng(config.random_state)
    (mean, free_mask, free_mean, free_covariance, precision, logdet_covariance) = _free_space(model.priors)
    bounds = _free_bounds(config, free_mask)
    d = int(np.sum(free_mask))

    # -----------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------
    def evaluate_free(x):
        """Evaluate the true negative log joint."""
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("optimizer proposed non-finite parameter values")
        full = _reconstruct(x, mean, free_mask)
        out = model.evaluate(full, subject_data)
        ll_vec = np.asarray(out["loglik"], dtype=float)
        if not np.all(np.isfinite(ll_vec)):
            raise FloatingPointError("model returned a non-finite log likelihood")
        loglik = float(np.sum(ll_vec))
        logprior = _log_prior_free(x, free_mean, precision, logdet_covariance)
        if not np.isfinite(logprior):
            raise FloatingPointError("prior returned a non-finite log density")
        logjoint = loglik + logprior
        if not np.isfinite(logjoint):
            raise FloatingPointError("non-finite log joint")
        return (-logjoint, loglik, logprior)

    # -----------------------------------------------------------------
    # All parameters fixed
    # -----------------------------------------------------------------
    if d == 0:
        full = mean.copy()
        out = model.evaluate(full, subject_data)
        ll_vec = np.asarray(out["loglik"], dtype=float)
        if not np.all(np.isfinite(ll_vec)):
            raise FloatingPointError("model returned a non-finite log likelihood")
        ll = float(np.sum(ll_vec))
        diagnostics = OptimizationDiagnostics(
            starts=[],
            search_path=np.zeros((1, 0), dtype=float),
            search_log_joint=np.array([ll], dtype=float),
            lbfgsb_success=True,
            lbfgsb_status=0,
            lbfgsb_message=("all static parameters fixed"),
            abs_grad=0.0,
            at_hard_bounds=np.zeros(model.n_parameters, dtype=bool),
            hess_method=config.hessian_method,
            hess_raw_min_eig=np.nan,
            hess_raw_max_eig=np.nan,
            hess_condition_number=np.nan,
            hess_ill_conditioned=False,
            laplace_valid=True,
            laplace_fragile=False,
        )

        return OptimizationResult(
            parameters=full,
            free_parameters=np.zeros(0, dtype=float),
            log_likelihood=ll,
            log_prior=0.0,
            log_joint=ll,
            hessian=np.zeros((0, 0), dtype=float),
            covariance=np.zeros((0, 0), dtype=float),
            log_evidence=ll,
            diagnostics=diagnostics,
            free_mask=free_mask,
        )

    # -----------------------------------------------------------------
    # Generate initialization points
    # -----------------------------------------------------------------
    init_list = [_clip_to_bounds(free_mean, bounds)]

    for _ in range(config.num_init - 1):
        init = rng.multivariate_normal(free_mean, free_covariance)
        init = _clip_to_bounds(init, bounds)
        init_list.append(init)

    # -----------------------------------------------------------------
    # Multi-start L-BFGS-B
    # -----------------------------------------------------------------
    starts: list[StartRecord] = []

    best = None
    best_fval = np.inf

    best_path = None
    best_path_log_joint = None

    for init in init_list:
        path = [np.asarray(init, dtype=float).copy()]

        def objective(x):
            return evaluate_free(x)[0]

        def callback(xk):
            path.append(np.asarray(xk, dtype=float).copy())

        res = minimize(objective, init, method="L-BFGS-B", bounds=bounds, callback=callback, options={"maxiter": config.maxiter})
        fval, ll, lp = evaluate_free(res.x)
        logjoint = -float(fval)
        gradient = np.asarray(getattr(res, "jac", np.full(d, np.nan)), dtype=float)

        if np.all(np.isfinite(gradient)):
            gradient_norm = float(np.linalg.norm(gradient))
        else:
            gradient_norm = np.nan

        full_final = _reconstruct(res.x, mean, free_mask)

        starts.append(
            StartRecord(
                initial_parameters=_reconstruct(init, mean, free_mask),
                final_parameters=full_final,
                log_joint=logjoint,
                success=bool(res.success and np.isfinite(logjoint)),
                status=int(res.status),
                message=str(res.message),
                n_iter=int(getattr(res, "nit", 0)),
                gradient_norm=gradient_norm,
            )
        )

        if np.isfinite(fval) and fval < best_fval:
            best = res
            best_fval = float(fval)

            best_path = np.vstack(path)

            # Re-evaluate the true objective along the winning path.
            best_path_log_joint = np.array([-evaluate_free(z)[0] for z in best_path], dtype=float)

    # -----------------------------------------------------------------
    # Verify MAP result
    # -----------------------------------------------------------------
    if best is None or not np.isfinite(best_fval):
        raise RuntimeError("MAP optimization did not find a finite objective value")
    xhat = np.asarray(best.x, dtype=float)
    full_hat = _reconstruct(xhat, mean, free_mask)
    (final_f, final_ll, final_lp) = evaluate_free(xhat)
    logjoint = -float(final_f)

    # -----------------------------------------------------------------
    # Detect hard-bound MAP solutions
    # -----------------------------------------------------------------
    at_bounds_full = _at_hard_bounds(full_hat, config.hard_bounds)
    at_any_bound = bool(np.any(at_bounds_full))

    # -----------------------------------------------------------------
    # Independent observed Hessian
    # -----------------------------------------------------------------
    def clean_objective(x):
        value, _, _ = evaluate_free(x)
        return value

    H = central_hessian(clean_objective, xhat, config.hessian_relative_step)
    if not np.all(np.isfinite(H)):
        raise FloatingPointError("observed Hessian contains non-finite values")
    eig = np.linalg.eigvalsh(H)
    min_eig = float(np.min(eig))
    max_eig = float(np.max(eig))
    positive_definite = bool(np.all(eig > 0.0))
    if positive_definite and min_eig > 0:
        condition_number = float(max_eig / min_eig)
    else:
        condition_number = np.inf
    hessian_ill_conditioned = bool(positive_definite and (not np.isfinite(condition_number) or condition_number >= config.condition_number_warn))
    laplace_valid = bool(positive_definite)

    # A hard-bound MAP solution can still have a numerically positive
    # Hessian, but the ordinary unconstrained Gaussian Laplace
    # approximation is then potentially unreliable.
    laplace_fragile = bool(laplace_valid and (hessian_ill_conditioned or at_any_bound))

    if hessian_ill_conditioned and config.verbose:
        warnings.warn("observed Hessian is ill-conditioned; Laplace uncertainty/evidence may be unreliable", RuntimeWarning, stacklevel=2)

    if at_any_bound and config.verbose:
        warnings.warn(
            "MAP estimate lies on a hard parameter bound; the standard Gaussian Laplace approximation may be unreliable", RuntimeWarning, stacklevel=2
        )

    # -----------------------------------------------------------------
    # Laplace covariance and evidence
    # -----------------------------------------------------------------
    if laplace_valid:
        covariance = np.linalg.inv(H)
        sign, logdet_hessian = np.linalg.slogdet(H)
        if sign > 0:
            log_evidence = float(logjoint + 0.5 * d * np.log(2.0 * np.pi) - 0.5 * logdet_hessian)
        else:
            log_evidence = np.nan
    else:
        covariance = None
        log_evidence = np.nan

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------
    best_gradient = np.asarray(best.jac, dtype=float)

    if np.all(np.isfinite(best_gradient)):
        best_gradient_norm = float(np.linalg.norm(best_gradient))
    else:
        best_gradient_norm = np.nan

    diagnostics = OptimizationDiagnostics(
        starts=starts,
        search_path=np.asarray([_reconstruct(x, mean, free_mask) for x in best_path], dtype=float),
        search_log_joint=best_path_log_joint,
        lbfgsb_success=bool(best.success),
        lbfgsb_status=int(best.status),
        lbfgsb_message=str(best.message),
        abs_grad=best_gradient_norm,
        at_hard_bounds=at_bounds_full,
        hess_method=config.hessian_method,
        hess_raw_min_eig=min_eig,
        hess_raw_max_eig=max_eig,
        hess_condition_number=condition_number,
        hess_ill_conditioned=hessian_ill_conditioned,
        laplace_valid=laplace_valid,
        laplace_fragile=laplace_fragile,
    )

    # -----------------------------------------------------------------
    # Result
    # -----------------------------------------------------------------

    return OptimizationResult(
        parameters=full_hat,
        free_parameters=xhat,
        log_likelihood=float(final_ll),
        log_prior=float(final_lp),
        log_joint=logjoint,
        hessian=H,
        covariance=covariance,
        log_evidence=log_evidence,
        diagnostics=diagnostics,
        free_mask=free_mask,
    )
