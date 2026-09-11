"""MAP optimization and independent observed-Hessian Laplace inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import warnings
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Config:
    """Individual-fit configuration.

    ``latent_samples`` and ``latent_interval`` are retained only when
    ``latent_uncertainty='propagated'``; for all other modes they are resolved
    to ``None`` so they cannot affect computation or metadata.
    """

    num_init: int = 5
    random_state: Optional[int] = 42
    verbose: bool = True
    display: bool = False
    maxiter: int = 1000
    hard_bounds: Optional[Sequence[Optional[tuple[Optional[float], Optional[float]]]]] = None
    hessian_method: str = "central_fd"
    hessian_relative_step: float = 1e-4
    condition_number_warn: float = 1e12
    latent_uncertainty: str = "none"
    latent_samples: Optional[int] = 1000
    latent_interval: Optional[float] = 0.95

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


@dataclass
class RuntimeSanityTracker:
    nonfinite_state: int = 0
    invalid_probability: int = 0
    invalid_process_covariance: int = 0
    invalid_observation_covariance: int = 0
    filter_update_failure: int = 0
    nonfinite_loglik: int = 0
    other_domain_error: int = 0

    @property
    def total(self) -> int:
        return int(sum(vars(self).values()))

    def record(self, message: str):
        msg = message.lower()
        if "state" in msg and ("finite" in msg or "covariance" in msg):
            self.nonfinite_state += 1
        elif "probab" in msg or "categorical" in msg or "bernoulli" in msg:
            self.invalid_probability += 1
        elif "process_covariance" in msg or "process covariance" in msg:
            self.invalid_process_covariance += 1
        elif "observation_covariance" in msg or "observation covariance" in msg:
            self.invalid_observation_covariance += 1
        elif "filter" in msg or "curvature" in msg:
            self.filter_update_failure += 1
        elif "log likelihood" in msg or "loglik" in msg:
            self.nonfinite_loglik += 1
        else:
            self.other_domain_error += 1


@dataclass
class StartRecord:
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
    starts: list[StartRecord] = field(default_factory=list)
    search_path: Optional[np.ndarray] = None
    search_log_joint: Optional[np.ndarray] = None
    lbfgsb_success: bool = False
    lbfgsb_status: Optional[int] = None
    lbfgsb_message: str = ""
    abs_grad: float = np.nan
    n_invalid_evaluations: int = 0
    runtime_sanity: Optional[RuntimeSanityTracker] = None
    at_hard_bounds: Optional[np.ndarray] = None
    hess_method: str = "central_fd"
    hess_raw_min_eig: float = np.nan
    hess_raw_max_eig: float = np.nan
    hess_condition_number: float = np.nan
    hess_ill_conditioned: bool = False
    laplace_valid: bool = False
    laplace_fragile: bool = False


@dataclass
class OptimizationResult:
    parameters: np.ndarray
    free_parameters: np.ndarray
    log_likelihood: float
    log_prior: float
    log_joint: float
    hessian: np.ndarray
    covariance: Optional[np.ndarray]
    log_evidence: float
    diagnostics: OptimizationDiagnostics
    free_mask: np.ndarray


def _free_space(priors):
    mean = np.asarray(priors.mean, dtype=float)
    cov = np.asarray(priors.covariance, dtype=float)
    fixed = np.isclose(np.diag(cov), 0.0, atol=1e-14, rtol=0.0)
    free = ~fixed
    free_mean = mean[free]
    free_cov = cov[np.ix_(free, free)]
    if free_cov.size:
        eig = np.linalg.eigvalsh(free_cov)
        if np.min(eig) <= 0:
            raise ValueError("free-parameter prior covariance must be positive definite")
        precision = np.linalg.inv(free_cov)
        sign, logdet = np.linalg.slogdet(free_cov)
        if sign <= 0:
            raise ValueError("free-parameter prior covariance must be positive definite")
        logdet_cov = float(logdet)
    else:
        precision = np.zeros((0, 0))
        logdet_cov = 0.0
    return mean, cov, free, free_mean, free_cov, precision, logdet_cov


def _reconstruct(free_values, mean, free_mask):
    full = mean.copy()
    full[free_mask] = np.asarray(free_values, dtype=float)
    return full


def _log_prior_free(x, mean, precision, logdet_cov):
    d = len(mean)
    if d == 0:
        return 0.0
    delta = np.asarray(x, dtype=float) - mean
    return float(-0.5 * (d * np.log(2 * np.pi) + logdet_cov + delta @ precision @ delta))


def _free_bounds(config: Config, free_mask):
    if config.hard_bounds is None:
        return None
    out = []
    for is_free, pair in zip(free_mask, config.hard_bounds):
        if not is_free:
            continue
        out.append((None, None) if pair is None else tuple(pair))
    return out


def _inside_bounds(x, bounds):
    if bounds is None:
        return True
    for value, (lo, hi) in zip(x, bounds):
        if lo is not None and value < lo:
            return False
        if hi is not None and value > hi:
            return False
    return True


def _clip_to_bounds(x, bounds):
    if bounds is None:
        return x
    x = np.asarray(x, dtype=float).copy()
    for j, (lo, hi) in enumerate(bounds):
        if lo is not None:
            x[j] = max(x[j], lo + 1e-10)
        if hi is not None:
            x[j] = min(x[j], hi - 1e-10)
    return x


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
        raise ValueError("objective is non-finite at MAP during Hessian calculation")
    for i in range(d):
        xp = x.copy(); xp[i] += h[i]
        xm = x.copy(); xm[i] -= h[i]
        fp, fm = float(func(xp)), float(func(xm))
        if not np.isfinite(fp) or not np.isfinite(fm):
            raise ValueError("non-finite objective near MAP during Hessian calculation")
        H[i, i] = (fp - 2.0 * f0 + fm) / (h[i] ** 2)
        for j in range(i + 1, d):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            vals = [float(func(z)) for z in (xpp, xpm, xmp, xmm)]
            if not np.all(np.isfinite(vals)):
                raise ValueError("non-finite objective near MAP during Hessian calculation")
            Hij = (vals[0] - vals[1] - vals[2] + vals[3]) / (4.0 * h[i] * h[j])
            H[i, j] = H[j, i] = Hij
    return 0.5 * (H + H.T)


def optimize_map(subject_data, model, config: Config, *, rng=None) -> OptimizationResult:
    """Fit one subject by multi-start L-BFGS-B and independent Hessian recomputation."""
    if rng is None:
        rng = np.random.default_rng(config.random_state)
    mean, cov, free_mask, free_mean, free_cov, precision, logdet_cov = _free_space(model.priors)
    bounds = _free_bounds(config, free_mask)
    tracker = RuntimeSanityTracker()
    invalid_penalty = 1e100

    def evaluate_free(x, *, track=True):
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)) or not _inside_bounds(x, bounds):
            if track:
                tracker.other_domain_error += 1
            return invalid_penalty, np.nan, np.nan
        full = _reconstruct(x, mean, free_mask)
        try:
            out = model.evaluate(full, subject_data)
            ll_vec = np.asarray(out["loglik"], dtype=float)
            if not np.all(np.isfinite(ll_vec)):
                raise ValueError("non-finite log likelihood")
            loglik = float(np.sum(ll_vec))
            logprior = _log_prior_free(x, free_mean, precision, logdet_cov)
            logjoint = loglik + logprior
            if not np.isfinite(logjoint):
                raise ValueError("non-finite log likelihood or log prior")
            return -logjoint, loglik, logprior
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            if track:
                tracker.record(str(exc))
            return invalid_penalty, np.nan, np.nan

    d = int(np.sum(free_mask))
    starts: list[StartRecord] = []
    best = None
    best_path = None
    best_path_f = None

    if d == 0:
        full = mean.copy()
        out = model.evaluate(full, subject_data)
        ll = float(np.sum(out["loglik"]))
        diag = OptimizationDiagnostics(
            starts=[], search_path=np.zeros((1, 0)), search_log_joint=np.array([ll]),
            lbfgsb_success=True, lbfgsb_status=0, lbfgsb_message="all static parameters fixed",
            abs_grad=0.0, n_invalid_evaluations=0, runtime_sanity=tracker,
            at_hard_bounds=np.zeros(model.n_parameters, dtype=bool), hess_method=config.hessian_method,
            hess_raw_min_eig=np.inf, hess_raw_max_eig=np.inf, hess_condition_number=1.0,
            hess_ill_conditioned=False, laplace_valid=True, laplace_fragile=False,
        )
        return OptimizationResult(
            parameters=full, free_parameters=np.zeros(0), log_likelihood=ll, log_prior=0.0,
            log_joint=ll, hessian=np.zeros((0, 0)), covariance=np.zeros((0, 0)),
            log_evidence=ll, diagnostics=diag, free_mask=free_mask,
        )

    init_list = [free_mean.copy()]
    for _ in range(config.num_init - 1):
        init = rng.multivariate_normal(free_mean, free_cov)
        init_list.append(_clip_to_bounds(init, bounds))

    for init in init_list:
        path = [np.asarray(init, dtype=float).copy()]
        path_f = []

        def objective(x):
            return evaluate_free(x, track=True)[0]

        def callback(xk):
            path.append(np.asarray(xk, dtype=float).copy())
            f, _, _ = evaluate_free(xk, track=False)
            path_f.append(-float(f))

        res = minimize(
            objective,
            init,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={"maxiter": config.maxiter},
        )
        fval, ll, lp = evaluate_free(res.x, track=False)
        logjoint = -float(fval)
        grad = np.asarray(getattr(res, "jac", np.full(d, np.nan)), dtype=float)
        grad_norm = float(np.linalg.norm(grad)) if np.all(np.isfinite(grad)) else np.nan
        full_final = _reconstruct(res.x, mean, free_mask)
        starts.append(
            StartRecord(
                initial_parameters=_reconstruct(init, mean, free_mask),
                final_parameters=full_final,
                log_joint=logjoint,
                success=bool(res.success and np.isfinite(logjoint) and fval < invalid_penalty),
                status=int(res.status),
                message=str(res.message),
                n_iter=int(getattr(res, "nit", 0)),
                gradient_norm=grad_norm,
            )
        )
        if best is None or fval < best.fun:
            best = res
            best.fun = fval
            best._ll = ll
            best._lp = lp
            best_path = np.vstack(path)
            # callback excludes initial/final in some scipy versions; recompute cleanly.
            best_path_f = np.array([-evaluate_free(z, track=False)[0] for z in best_path])

    if best is None or not np.isfinite(best.fun) or best.fun >= invalid_penalty:
        raise RuntimeError("MAP optimization did not find a finite objective value")

    xhat = np.asarray(best.x, dtype=float)
    full_hat = _reconstruct(xhat, mean, free_mask)
    final_f, final_ll, final_lp = evaluate_free(xhat, track=False)
    logjoint = -float(final_f)

    # Crucial separation: recompute observed Hessian independently from L-BFGS-B.
    def clean_objective(x):
        value, _, _ = evaluate_free(x, track=False)
        return value

    H = central_hessian(clean_objective, xhat, config.hessian_relative_step)
    eig = np.linalg.eigvalsh(H)
    min_eig = float(np.min(eig))
    max_eig = float(np.max(eig))
    pd = bool(np.all(eig > 0.0))
    cond = float(max_eig / min_eig) if pd and min_eig > 0 else np.inf
    ill = bool(pd and (not np.isfinite(cond) or cond >= config.condition_number_warn))
    laplace_valid = pd and np.all(np.isfinite(H))
    laplace_fragile = bool(laplace_valid and ill)
    covariance = np.linalg.inv(H) if laplace_valid else None
    if laplace_valid:
        sign, logdetH = np.linalg.slogdet(H)
        log_evidence = float(logjoint + 0.5 * d * np.log(2 * np.pi) - 0.5 * logdetH) if sign > 0 else np.nan
    else:
        log_evidence = np.nan

    at_bounds_full = np.zeros(model.n_parameters, dtype=bool)
    if config.hard_bounds is not None:
        for j, (value, pair) in enumerate(zip(full_hat, config.hard_bounds)):
            if pair is None:
                continue
            lo, hi = pair
            tol = 1e-7 * max(1.0, abs(float(value)))
            at_bounds_full[j] = (lo is not None and abs(value - lo) <= tol) or (hi is not None and abs(value - hi) <= tol)

    diag = OptimizationDiagnostics(
        starts=starts,
        search_path=np.asarray([_reconstruct(x, mean, free_mask) for x in best_path]),
        search_log_joint=best_path_f,
        lbfgsb_success=bool(best.success),
        lbfgsb_status=int(best.status),
        lbfgsb_message=str(best.message),
        abs_grad=float(np.linalg.norm(np.asarray(best.jac, dtype=float))) if np.all(np.isfinite(best.jac)) else np.nan,
        n_invalid_evaluations=tracker.total,
        runtime_sanity=tracker,
        at_hard_bounds=at_bounds_full,
        hess_method=config.hessian_method,
        hess_raw_min_eig=min_eig,
        hess_raw_max_eig=max_eig,
        hess_condition_number=cond,
        hess_ill_conditioned=ill,
        laplace_valid=laplace_valid,
        laplace_fragile=laplace_fragile,
    )

    return OptimizationResult(
        parameters=full_hat,
        free_parameters=xhat,
        log_likelihood=float(final_ll),
        log_prior=float(final_lp),
        log_joint=logjoint,
        hessian=H,
        covariance=covariance,
        log_evidence=log_evidence,
        diagnostics=diag,
        free_mask=free_mask,
    )
