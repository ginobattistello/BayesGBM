"""Tutorial 07 — prior and posterior predictive checks.

Predictive checks generate replicated *observations*. They are different from
propagated latent uncertainty, which summarizes uncertainty over latent states.
"""
import numpy as np
from bayesgbm import Config, individual_fit, prior_predictive, posterior_predictive
from tutorial_models import binary_learning_model, simulate_binary_subject

rng = np.random.default_rng(42)
data = [simulate_binary_subject(rng, n_trials=60)]
model = binary_learning_model()

prior = prior_predictive(model, data[0], n_samples=100, random_state=42)
fit = individual_fit(data, model, config=Config(num_init=4, random_state=42, verbose=False))
posterior = posterior_predictive(fit, subject=0, n_samples=100, random_state=42)

obs_rate = np.mean(data[0]["y"])
prior_rates = np.array([np.mean(d["y"]) for d in prior.replicated_data])
post_rates = np.array([np.mean(d["y"]) for d in posterior.replicated_data])
print(f"observed P(choice=1): {obs_rate:.3f}")
print(f"prior predictive mean: {prior_rates.mean():.3f}")
print(f"posterior predictive mean: {post_rates.mean():.3f}")
