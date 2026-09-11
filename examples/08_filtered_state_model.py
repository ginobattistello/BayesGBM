"""Tutorial 08 — internal latent-state uncertainty from filtering.

Here the hidden state itself is uncertain because both P0 and process
covariance Q are non-zero. Filtering is therefore part of the likelihood at
every optimizer evaluation. `latent_uncertainty='filtered'` asks BayesGBM to
retain P_t for plotting/reporting; it does not create the stochastic model.
"""
import numpy as np
from bayesgbm import Config, individual_fit
from tutorial_models import filtered_continuous_model

rng = np.random.default_rng(42)
T = 50
# Simulate a simple AR(1) hidden state with process and observation noise.
x = 0.0
y = np.zeros(T)
for t in range(T):
    y[t] = x + rng.normal(0, .5)
    x = .8 * x + rng.normal(0, np.sqrt(.10))

data = [{"y": y, "u": None}]
fit = individual_fit(
    data,
    filtered_continuous_model(),
    config=Config(num_init=4, random_state=42, latent_uncertainty="filtered", display=False),
)
print(fit.summary())
fit.plot(subject=0)
