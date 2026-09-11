"""Tutorial 02 — continuous Gaussian outcomes.

A Gaussian observation model needs both a predicted mean and a residual
observation covariance R. Here R=0.25 is known and fixed for clarity.
"""
import numpy as np
from bayesgbm import Config, individual_fit
from tutorial_models import continuous_model, simulate_continuous_subject

rng = np.random.default_rng(42)
data = [simulate_continuous_subject(rng) for _ in range(3)]
fit = individual_fit(
    data,
    continuous_model(),
    config=Config(num_init=4, random_state=42, display=False),
)
print(fit.summary(subject=0))
fit.plot(subject=0)
