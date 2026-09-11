"""Tutorial 04 — hierarchical group refitting.

The current BayesGBM HBI entry point is a compact hierarchical empirical-Bayes
Laplace approximation inspired by CBM/HBI. It repeatedly refits subjects under
responsibility-weighted Gaussian group priors.
"""
import numpy as np
from bayesgbm import Config, HBIConfig, hbi_main
from tutorial_models import binary_learning_model, simulate_binary_subject

rng = np.random.default_rng(42)
data = [simulate_binary_subject(rng, n_trials=50) for _ in range(6)]
model = binary_learning_model()

result = hbi_main(
    data,
    [model],
    fit_config=Config(num_init=2, random_state=42, verbose=False),
    hbi_config=HBIConfig(maxiter=5, tol=1e-2, verbose=True),
)
print("group evolution mean:", result.group_priors[0].evolution.mean)
print("group observation mean:", result.group_priors[0].observation.mean)
print("model frequency:", result.model_frequency)
