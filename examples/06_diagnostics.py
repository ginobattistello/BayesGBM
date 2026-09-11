"""Tutorial 06 — convergence, curvature, information, identifiability."""
import numpy as np
from bayesgbm import (
    Config, individual_fit,
    convergence_diagnostics, posterior_hessian_diagnostics,
    prior_preconditioned_information, numerical_local_identifiability,
)
from tutorial_models import binary_learning_model, simulate_binary_subject

rng = np.random.default_rng(42)
data = [simulate_binary_subject(rng, n_trials=70)]
fit = individual_fit(data, binary_learning_model(), config=Config(num_init=5, random_state=42, verbose=False))

print("\nConvergence\n", convergence_diagnostics(fit))
print("\nObserved-Hessian diagnostics\n", posterior_hessian_diagnostics(fit))
print("\nPrior-preconditioned information\n", prior_preconditioned_information(fit))
print("\nNumerical local identifiability\n", numerical_local_identifiability(fit))
