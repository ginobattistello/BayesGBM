"""Small model/simulation helpers shared by the BayesGBM tutorials."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from bayesgbm import GaussianPrior, Priors, StateModel


def sigmoid(z):
    return expit(z)


def binary_learning_model(*, fixed_alpha=False):
    """Two-option Rescorla-Wagner model with Bernoulli choices."""
    def evolution(x, theta, u_t, y_t):
        alpha = sigmoid(theta[0])
        x = x.copy()
        choice = int(y_t)
        reward = float(u_t["reward"])
        x[choice] += alpha * (reward - x[choice])
        return x

    def observation(x, phi, u_t):
        beta = np.exp(phi[0])
        return sigmoid(beta * (x[1] - x[0]))

    return StateModel(
        evolution=evolution,
        observation=observation,
        family="bernoulli",
        priors=Priors(
            GaussianPrior([0.0], [0.0 if fixed_alpha else 1.0], names=["alpha_raw"]),
            GaussianPrior([1.0], [1.0], names=["log_beta"]),
        ),
        initial_state=[0.5, 0.5],
        state_names=["Q0", "Q1"],
        name="two-option learning",
    )


def simulate_binary_subject(rng, n_trials=80, alpha=.30, beta=3.0):
    reward_probability = np.array([.75, .25])
    q = np.array([.5, .5])
    y = np.zeros(n_trials, dtype=int)
    reward = np.zeros(n_trials)
    for t in range(n_trials):
        p1 = sigmoid(beta * (q[1] - q[0]))
        choice = int(rng.random() < p1)
        r = float(rng.random() < reward_probability[choice])
        y[t], reward[t] = choice, r
        q[choice] += alpha * (r - q[choice])
    return {"y": y, "u": {"reward": reward}}


def continuous_model():
    """Deterministic one-state linear dynamics with Gaussian observations."""
    def evolution(x, theta, u_t, y_t):
        return np.array([theta[0] * x[0] + float(u_t)])

    def observation(x, phi, u_t):
        return np.array([phi[0] + x[0]])

    return StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(
            GaussianPrior([.7], [.5], names=["state_gain"]),
            GaussianPrior([0.0], [1.0], names=["offset"]),
        ),
        initial_state=[0.0],
        observation_covariance=.25,  # known residual variance for this tutorial
        state_names=["x"],
        name="continuous state model",
    )


def simulate_continuous_subject(rng, n_trials=60, gain=.7, offset=.2, sigma=.5):
    u = rng.normal(0, .3, n_trials)
    x = 0.0
    y = np.zeros(n_trials)
    for t in range(n_trials):
        y[t] = offset + x + rng.normal(0, sigma)
        x = gain * x + u[t]
    return {"y": y, "u": u}


def filtered_continuous_model():
    """Linear-Gaussian hidden state with genuine internal state uncertainty."""
    def evolution(x, theta, u_t, y_t):
        return np.array([theta[0] * x[0]])

    def observation(x, phi, u_t):
        return np.array([x[0] + phi[0]])

    return StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(
            GaussianPrior([.8], [.2], names=["persistence"]),
            GaussianPrior([0.0], [1.0], names=["offset"]),
        ),
        initial_state=[0.0],
        initial_state_covariance=[1.0],
        process_covariance=[.10],
        observation_covariance=[.25],
        state_names=["hidden_state"],
        name="stochastic continuous state model",
    )
