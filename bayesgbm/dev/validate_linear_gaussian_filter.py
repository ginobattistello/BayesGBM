"""Reference check: generalized Gaussian filter equals a scalar Kalman filter."""
import jax.numpy as jnp
import numpy as np

from bayesgbm import GaussianPrior, Priors, StateModel

A, Q, R, P0 = 0.8, 0.10, 0.25, 1.0


def evolution(x, theta, u_t, y_t):
    return jnp.asarray([theta[0] * x[0]])


def observation(x, phi, u_t):
    return x[0]


model = StateModel(
    evolution=evolution,
    observation=observation,
    family="gaussian",
    priors=Priors(GaussianPrior([A], [0.0]), GaussianPrior([0.0], [0.0])),
    initial_state=[0.0],
    initial_state_covariance=[P0],
    process_covariance=[Q],
    observation_covariance=[R],
)

rng = np.random.default_rng(4)
y = rng.normal(size=25)
data = {"y": y, "u": None}
run = model.evaluate(model.parameter_layout.mean, data)

m, P = 0.0, P0
ms, Ps, lls = [], [], []
for yt in y:
    S = P + R
    lls.append(-0.5 * (np.log(2 * np.pi * S) + (yt - m) ** 2 / S))
    K = P / S
    mf = m + K * (yt - m)
    Pf = (1 - K) * P
    ms.append(mf); Ps.append(Pf)
    m = A * mf
    P = A * A * Pf + Q

np.testing.assert_allclose(run["states"][:, 0], ms, rtol=1e-7, atol=1e-7)
np.testing.assert_allclose(run["state_covariance"][:, 0, 0], Ps, rtol=1e-7, atol=1e-7)
np.testing.assert_allclose(run["loglik"], lls, rtol=1e-7, atol=1e-7)
print("PASS")
