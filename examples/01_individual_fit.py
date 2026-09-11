"""Tutorial 01 — individual fit and propagated latent uncertainty.

The latent Q values are deterministic once static parameters are known.
BayesGBM therefore samples the final Laplace posterior over static parameters
and reruns the state recursion. The shadow is the empirical 95% credible band
of the resulting Q trajectories.

This example also includes a small parameter-recovery demonstration:
subjects are generated with different alpha and beta values, fitted
independently, and the generating and estimated values are compared.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from tutorial_models import binary_learning_model, simulate_binary_subject

from bayesgbm import Config, individual_fit

# ---------------------------------------------------------------------
# 1. Simulate subjects with different generating parameters
# ---------------------------------------------------------------------
rng = np.random.default_rng(42)
n_subjects = 20

true_alpha = rng.uniform(0.10, 0.60, size=n_subjects)
true_beta = rng.uniform(1.0, 5.0, size=n_subjects)

data = [simulate_binary_subject(rng, n_trials=150, alpha=alpha, beta=beta) for alpha, beta in zip(true_alpha, true_beta)]


# ---------------------------------------------------------------------
# 2. Define and fit the model
# ---------------------------------------------------------------------
model = binary_learning_model()

fit = individual_fit(
    data, model, config=Config(num_init=5, random_state=42, latent_uncertainty="propagated", latent_samples=500, latent_interval=0.95, display=False)
)


# ---------------------------------------------------------------------
# 3. Inspect one subject
# ---------------------------------------------------------------------
print(fit.summary(subject=0))
fit.plot(subject=0)


# ---------------------------------------------------------------------
# 4. Small parameter-recovery check
# ---------------------------------------------------------------------
# BayesGBM estimates unconstrained parameters:
#
#   alpha = sigmoid(alpha_raw)
#   beta  = exp(log_beta)
#
# Transform the MAP estimates back to their scientifically meaningful
# scale before comparing them with the generating parameters.
estimated_alpha = expit(fit.output.evolution_parameters[:, 0])
estimated_beta = np.exp(fit.output.observation_parameters[:, 0])


# ---------------------------------------------------------------------
# 5. Plot parameter recovery
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

parameters = [("Learning rate (alpha)", true_alpha, estimated_alpha), ("Inverse temperature (beta)", true_beta, estimated_beta)]

for ax, (label, true, estimated) in zip(axes, parameters):
    # Recovery statistics
    r = np.corrcoef(true, estimated)[0, 1]
    rmse = np.sqrt(np.mean((estimated - true) ** 2))

    # Individual subjects
    ax.scatter(true, estimated)

    # Identity line: perfect recovery
    lower = min(true.min(), estimated.min())
    upper = max(true.max(), estimated.max())

    ax.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True parameter")
    ax.set_ylabel("Estimated parameter")
    ax.set_title(label)

    ax.text(0.05, 0.95, f"r = {r:.2f}\nRMSE = {rmse:.2f}", transform=ax.transAxes, va="top")

fig.suptitle("Parameter recovery")
fig.tight_layout()

plt.show()
