import numpy as np
import pytest

from bayesgbm import GaussianPrior, Priors, StateModel


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@pytest.fixture
def binary_model():
    def evolution(x, theta, u_t, y_t):
        alpha = sigmoid(theta[0])
        x = x.copy()
        c = int(y_t)
        x[c] += alpha * (float(u_t["reward"]) - x[c])
        return x

    def observation(x, phi, u_t):
        beta = np.exp(phi[0])
        return sigmoid(beta * (x[1] - x[0]))

    return StateModel(
        evolution=evolution,
        observation=observation,
        family="bernoulli",
        priors=Priors(
            GaussianPrior([0.0], [1.0], names=["alpha_raw"]),
            GaussianPrior([1.0], [1.0], names=["log_beta"]),
        ),
        initial_state=[0.5, 0.5],
        state_names=["Q0", "Q1"],
        name="binary_rw",
    )


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    return [{
        "y": rng.integers(0, 2, 30),
        "u": {"reward": rng.integers(0, 2, 30).astype(float)},
    }]


@pytest.fixture
def gaussian_filter_model():
    def evolution(x, theta, u_t, y_t):
        return np.array([theta[0] * x[0]])

    def observation(x, phi, u_t):
        return np.array([x[0] + phi[0]])

    return StateModel(
        evolution=evolution,
        observation=observation,
        family="gaussian",
        priors=Priors(
            GaussianPrior([0.8], [0.2], names=["a"]),
            GaussianPrior([0.0], [1.0], names=["offset"]),
        ),
        initial_state=[0.0],
        initial_state_covariance=[1.0],
        process_covariance=[0.1],
        observation_covariance=[0.25],
        state_names=["x"],
        name="linear_gaussian",
    )
