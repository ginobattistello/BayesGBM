"""Tutorial 02 — continuous Gaussian outcomes with noise recovery.

This example fits a deterministic latent-state model with Gaussian
observations and performs a small parameter-recovery demonstration.

Subjects are simulated with different:

- state gain
- observation offset
- observation noise sigma

The observation noise is estimated through log_sigma so that the
Gaussian observation variance remains strictly positive.
"""

import matplotlib.pyplot as plt
import numpy as np
from tutorial_models import continuous_model, simulate_continuous_subject

from bayesgbm import Config, individual_fit

# ---------------------------------------------------------------------
# 1. Simulate subjects with different generating parameters
# ---------------------------------------------------------------------
rng = np.random.default_rng(42)

n_subjects = 30
n_trials = 80

true_gain = rng.uniform(0.40, 0.90, size=n_subjects)
true_offset = rng.uniform(-0.50, 0.50, size=n_subjects)
true_sigma = rng.uniform(0.20, 1.00, size=n_subjects)

data = [
    simulate_continuous_subject(rng, n_trials=n_trials, gain=gain, offset=offset, sigma=sigma)
    for gain, offset, sigma in zip(true_gain, true_offset, true_sigma)
]


# ---------------------------------------------------------------------
# 2. Fit the model
# ---------------------------------------------------------------------
model = continuous_model()

print("n_theta:", model.n_theta)
print("n_phi:", model.n_phi)
print("n_parameters:", model.n_parameters)
print("parameter names:", model.priors.names)
print("prior mean:", model.priors.mean)
print("prior covariance:")
print(model.priors.covariance)

fit = individual_fit(data, model, config=Config(num_init=5, random_state=42, display=False))


# ---------------------------------------------------------------------
# 3. Inspect one subject
# ---------------------------------------------------------------------
print(fit.summary(subject=0))
fit.plot(subject=0)


# ---------------------------------------------------------------------
# 4. Recover parameters
# ---------------------------------------------------------------------
estimated_gain = fit.output.evolution_parameters[:, 0]
estimated_offset = fit.output.observation_parameters[:, 0]

# BayesGBM estimates log_sigma.
# Transform back to the scientifically meaningful scale.
estimated_sigma = np.exp(fit.output.observation_parameters[:, 1])

print("\nParameter recovery")
print("------------------")

for s in range(n_subjects):
    print(
        f"Subject {s + 1:02d} | "
        f"gain: true={true_gain[s]:.3f}, "
        f"estimated={estimated_gain[s]:.3f} | "
        f"offset: true={true_offset[s]:.3f}, "
        f"estimated={estimated_offset[s]:.3f} | "
        f"sigma: true={true_sigma[s]:.3f}, "
        f"estimated={estimated_sigma[s]:.3f}"
    )


# ---------------------------------------------------------------------
# 5. Across-subject recovery statistics
# ---------------------------------------------------------------------
def recovery_statistics(true, estimated):
    r = np.corrcoef(true, estimated)[0, 1]
    rmse = np.sqrt(np.mean((estimated - true) ** 2))
    return r, rmse


r_gain, rmse_gain = recovery_statistics(true_gain, estimated_gain)
r_offset, rmse_offset = recovery_statistics(true_offset, estimated_offset)
r_sigma, rmse_sigma = recovery_statistics(true_sigma, estimated_sigma)


# ---------------------------------------------------------------------
# 6. Plot parameter recovery
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 4))

parameters = [
    ("State gain", true_gain, estimated_gain, r_gain, rmse_gain),
    ("Observation offset", true_offset, estimated_offset, r_offset, rmse_offset),
    ("Observation noise (sigma)", true_sigma, estimated_sigma, r_sigma, rmse_sigma),
]

for ax, (label, true, estimated, r, rmse) in zip(axes, parameters):
    # Individual subjects
    ax.scatter(true, estimated)

    # Plot limits shared between x and y
    lower = min(true.min(), estimated.min())
    upper = max(true.max(), estimated.max())
    span = upper - lower
    if span == 0:
        span = 1.0
    padding = 0.05 * span
    lower -= padding
    upper += padding

    # Identity line = perfect recovery
    ax.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True parameter")
    ax.set_ylabel("Estimated parameter")
    ax.set_title(label)
    ax.text(0.05, 0.95, (f"r = {r:.2f}\nRMSE = {rmse:.2f}"), transform=ax.transAxes, va="top")

fig.suptitle("Parameter recovery")
fig.tight_layout()

plt.show()
