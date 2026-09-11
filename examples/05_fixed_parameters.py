"""Tutorial 05 — fixing static parameters with zero prior variance.

`alpha_raw` is fixed exactly at its prior mean. It is removed from the free
optimization and Laplace dimension, while the full output still contains it.
"""
import numpy as np
from bayesgbm import Config, individual_fit
from tutorial_models import binary_learning_model, simulate_binary_subject

rng = np.random.default_rng(42)
data = [simulate_binary_subject(rng, n_trials=60)]
model = binary_learning_model(fixed_alpha=True)
fit = individual_fit(data, model, config=Config(num_init=3, random_state=42, verbose=False))
print(fit.summary())
print("full posterior covariance:\n", fit.math.covariance[0])
