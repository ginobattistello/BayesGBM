"""Tutorial 03 — random-effects Bayesian model selection.

Input is a subjects x models matrix of individual Laplace log evidences.
"""
import numpy as np
from bayesgbm import bms

rng = np.random.default_rng(42)
# Illustrative evidence: model 0 is better for most subjects.
lme = np.column_stack([
    rng.normal(-20, 2, 20),
    rng.normal(-23, 2, 20),
    rng.normal(-24, 2, 20),
])
result = bms(lme, n_samples=50_000, random_state=42)
print("model frequency:", np.round(result.model_frequency, 3))
print("exceedance probability:", np.round(result.exceedance_prob, 3))
print("protected exceedance probability:", np.round(result.protected_exceedance_prob, 3))
print("BOR:", round(result.bor, 4))
