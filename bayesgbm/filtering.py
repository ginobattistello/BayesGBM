"""Gaussian and Laplace-Gaussian state filtering for BayesGBM."""
from __future__ import annotations

import numpy as np

from .priors import covariance_matrix
from .state_model import trial_input


def _fd_jacobian(func, x, relative_step=1e-5):
    x = np.asarray(x, dtype=float).reshape(-1)
    f0 = np.asarray(func(x), dtype=float).reshape(-1)
    J = np.zeros((f0.size, x.size), dtype=float)
    for j in range(x.size):
        h = relative_step * max(1.0, abs(float(x[j])))
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        fp = np.asarray(func(xp), dtype=float).reshape(-1)
        fm = np.asarray(func(xm), dtype=float).reshape(-1)
        if fp.shape != f0.shape or fm.shape != f0.shape:
            raise ValueError("finite-difference function output changed shape")
        J[:, j] = (fp - fm) / (2.0 * h)
    if not np.all(np.isfinite(J)):
        raise ValueError("non-finite numerical Jacobian")
    return J


def _psd_support(P, tol=1e-12):
    P = 0.5 * (np.asarray(P, dtype=float) + np.asarray(P, dtype=float).T)
    vals, vecs = np.linalg.eigh(P)
    scale = max(1.0, float(np.max(np.abs(vals))) if vals.size else 1.0)
    if np.min(vals) < -tol * scale:
        raise ValueError("state covariance is not positive semidefinite")
    keep = vals > tol * scale
    if not np.any(keep):
        return np.zeros((P.shape[0], 0)), vals
    L = vecs[:, keep] * np.sqrt(vals[keep])[None, :]
    return L, vals


def _normal_logpdf(delta, cov):
    delta = np.asarray(delta, dtype=float).reshape(-1)
    cov = covariance_matrix(cov, delta.size, name="predictive covariance", allow_semidefinite=False)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("predictive covariance must be positive definite")
    return float(-0.5 * (delta.size * np.log(2 * np.pi) + logdet + delta @ np.linalg.solve(cov, delta)))


def _discrete_terms(model, family, x, phi, u_t, y_t, relative_step):
    p = np.asarray(model.observation(x, phi, u_t), dtype=float).reshape(-1)
    eps = 1e-12
    if family == "bernoulli":
        if p.size != 1:
            raise ValueError("Bernoulli observation must return one probability")
        prob = float(p[0])
        if not np.isfinite(prob) or prob < 0.0 or prob > 1.0:
            raise ValueError("invalid Bernoulli probability")
        prob = float(np.clip(prob, eps, 1.0 - eps))
        Jp = _fd_jacobian(
            lambda xx: np.asarray([model.observation(xx, phi, u_t)], dtype=float).reshape(-1),
            x,
            relative_step,
        )
        yv = float(y_t)
        loglik = yv * np.log(prob) + (1.0 - yv) * np.log1p(-prob)
        grad = Jp[0] * ((yv - prob) / (prob * (1.0 - prob)))
        fisher = np.outer(Jp[0], Jp[0]) / (prob * (1.0 - prob))
        return float(loglik), grad, fisher, prob

    if p.size < 2 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("invalid categorical probabilities")
    if not np.isclose(np.sum(p), 1.0, atol=1e-8, rtol=1e-7):
        raise ValueError("categorical probabilities must sum to one")
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p)
    yi = int(y_t)
    if yi < 0 or yi >= p.size:
        raise ValueError("categorical outcome outside probability vector")
    Jp = _fd_jacobian(
        lambda xx: np.asarray(model.observation(xx, phi, u_t), dtype=float).reshape(-1),
        x,
        relative_step,
    )
    loglik = np.log(p[yi])
    grad = Jp[yi] / p[yi]
    fisher = Jp.T @ (np.diag(1.0 / p) @ Jp)
    fisher = 0.5 * (fisher + fisher.T)
    return float(loglik), grad, fisher, p


def _discrete_update(
    model,
    family,
    m_pred,
    P_pred,
    phi,
    u_t,
    y_t,
    relative_step=1e-5,
    max_iter=30,
    tol=1e-9,
):
    """Laplace-Gaussian update in the uncertain subspace using Fisher curvature."""
    m_pred = np.asarray(m_pred, dtype=float).reshape(-1)
    L, _ = _psd_support(P_pred)
    if L.shape[1] == 0:
        ll, _, _, pred = _discrete_terms(model, family, m_pred, phi, u_t, y_t, relative_step)
        return m_pred.copy(), np.zeros_like(P_pred), ll, pred, 0, True

    z = np.zeros(L.shape[1], dtype=float)
    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        x = m_pred + L @ z
        ll, grad_x, fisher_x, _ = _discrete_terms(model, family, x, phi, u_t, y_t, relative_step)
        grad_z = L.T @ grad_x
        fisher_z = L.T @ fisher_x @ L
        A = np.eye(L.shape[1]) + fisher_z
        gpost = -z + grad_z
        try:
            step = np.linalg.solve(A, gpost)
        except np.linalg.LinAlgError as exc:
            raise ValueError("discrete filtering curvature is singular") from exc
        if not np.all(np.isfinite(step)):
            raise ValueError("non-finite discrete filtering update")
        z_new = z + step
        n_iter = it
        if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(z_new)):
            z = z_new
            converged = True
            break
        z = z_new

    x_mode = m_pred + L @ z
    ll, _, fisher_x, pred = _discrete_terms(model, family, x_mode, phi, u_t, y_t, relative_step)
    A = np.eye(L.shape[1]) + L.T @ fisher_x @ L
    sign, logdetA = np.linalg.slogdet(A)
    if sign <= 0:
        raise ValueError("discrete filtering posterior curvature is not positive definite")
    cov_z = np.linalg.inv(A)
    P_filt = L @ cov_z @ L.T
    P_filt = 0.5 * (P_filt + P_filt.T)
    # Laplace approximation to integral in support coordinates, whose prior is N(0, I).
    log_predictive = float(ll - 0.5 * (z @ z) - 0.5 * logdetA)
    return x_mode, P_filt, log_predictive, pred, n_iter, converged


def nonlinear_state_filter(
    model,
    parameters,
    subject_data,
    *,
    relative_step=1e-5,
    max_update_iter=30,
    update_tol=1e-9,
):
    """Run BayesGBM's filtering likelihood and return trial-wise state moments.

    Gaussian outcomes use an extended Kalman update. Bernoulli/categorical
    outcomes use a local Laplace-Gaussian update with Fisher curvature.
    """
    theta, phi = model.split_parameters(parameters)
    y = np.asarray(subject_data["y"])
    u = subject_data.get("u", None)
    T = len(y)
    n = model.n_state

    m_pred = model.initial_state.copy()
    P_pred = model.initial_covariance_matrix()

    pred_mean = np.zeros((T, n))
    pred_cov = np.zeros((T, n, n))
    filt_mean = np.zeros((T, n))
    filt_cov = np.zeros((T, n, n))
    loglik = np.zeros(T)
    predictions = []
    update_iterations = np.zeros(T, dtype=int)
    update_converged = np.ones(T, dtype=bool)

    for t in range(T):
        u_t = trial_input(u, t, T)
        pred_mean[t] = m_pred
        pred_cov[t] = P_pred

        if model.family == "gaussian":
            obs = np.asarray(model.observation(m_pred.copy(), phi.copy(), u_t), dtype=float).reshape(-1)
            G = _fd_jacobian(
                lambda xx: np.asarray(model.observation(xx, phi.copy(), u_t), dtype=float).reshape(-1),
                m_pred,
                relative_step,
            )
            R = model.observation_covariance_matrix(phi, u_t, obs.size)
            S = G @ P_pred @ G.T + R
            S = 0.5 * (S + S.T)
            yt = np.asarray(y[t], dtype=float).reshape(-1)
            if yt.shape != obs.shape:
                raise ValueError("Gaussian outcome and predicted mean dimensions differ")
            delta = yt - obs
            loglik[t] = _normal_logpdf(delta, S)
            K = P_pred @ G.T @ np.linalg.inv(S)
            m_filt = m_pred + K @ delta
            I = np.eye(n)
            # Joseph-form covariance update for numerical stability.
            P_filt = (I - K @ G) @ P_pred @ (I - K @ G).T + K @ R @ K.T
            P_filt = 0.5 * (P_filt + P_filt.T)
            predictions.append(obs)
            update_iterations[t] = 1
        else:
            m_filt, P_filt, ll, pred, nit, conv = _discrete_update(
                model,
                model.family,
                m_pred,
                P_pred,
                phi,
                u_t,
                y[t],
                relative_step=relative_step,
                max_iter=max_update_iter,
                tol=update_tol,
            )
            loglik[t] = ll
            predictions.append(pred)
            update_iterations[t] = nit
            update_converged[t] = conv

        filt_mean[t] = m_filt
        filt_cov[t] = P_filt

        # Predict next state from the filtered current state.
        next_mean = np.asarray(model.evolution(m_filt.copy(), theta.copy(), u_t, y[t]), dtype=float).reshape(-1)
        if next_mean.shape != m_filt.shape or not np.all(np.isfinite(next_mean)):
            raise ValueError("evolution must return a finite state with the same shape as x")
        F = _fd_jacobian(
            lambda xx: np.asarray(model.evolution(xx, theta.copy(), u_t, y[t]), dtype=float).reshape(-1),
            m_filt,
            relative_step,
        )
        Q = model.process_covariance_matrix(theta, u_t)
        next_cov = F @ P_filt @ F.T + Q
        next_cov = 0.5 * (next_cov + next_cov.T)
        _psd_support(next_cov)  # validation without modifying the covariance
        m_pred, P_pred = next_mean, next_cov

    pred_arr = np.asarray(predictions, dtype=float)
    if model.family in {"gaussian", "bernoulli"} and pred_arr.ndim == 2 and pred_arr.shape[1] == 1:
        pred_arr = pred_arr[:, 0]

    method = "extended_kalman_filter" if model.family == "gaussian" else "laplace_gaussian_fisher"
    return {
        "loglik": loglik,
        "states": filt_mean,
        "state_covariance": filt_cov,
        "predicted_state": pred_mean,
        "predicted_covariance": pred_cov,
        "prediction": pred_arr,
        "filtering_method": method,
        "update_iterations": update_iterations,
        "update_converged": update_converged,
    }
