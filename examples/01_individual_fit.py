"""Tutorial 01 — individual fit and propagated latent uncertainty.

The latent Q values are deterministic once static parameters are known.
BayesGBM therefore samples the final Laplace posterior over static parameters
and reruns the state recursion. The shadow is the empirical 95% credible band
of the resulting Q trajectories.
"""
import numpy as np
from bayesgbm import Config, individual_fit
from tutorial_models import binary_learning_model, simulate_binary_subject

rng = np.random.default_rng(42)
data = [simulate_binary_subject(rng) for _ in range(3)]
model = binary_learning_model()

fit = individual_fit(
    data,
    model,
    config=Config(
        num_init=5,
        random_state=42,
        latent_uncertainty="propagated",
        latent_samples=500,
        latent_interval=.95,
        display=False,
    ),
)

print(fit.summary(subject=0))
fit.plot(subject=0)
