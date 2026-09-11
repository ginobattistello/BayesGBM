# BayesGBM methods

## Generative state model

BayesGBM separates three quantities:

- `theta`: static evolution parameters;
- `phi`: static observation parameters;
- `x_t`: dynamic latent state.

A trial follows

```text
x_t -> observation(x_t, phi, u_t) -> likelihood(y_t)
    -> evolution(x_t, theta, u_t, y_t) -> x_{t+1}
```

The observation family is Gaussian, Bernoulli, or categorical. The modeller
returns a predicted mean, a Bernoulli probability, or a categorical probability
vector; BayesGBM constructs the likelihood.

## MAP and Laplace approximation

The static parameter posterior is optimized by multi-start L-BFGS-B. The
quasi-Newton matrix maintained by L-BFGS-B is not used for uncertainty or
model evidence. At the selected MAP, BayesGBM independently recomputes the
observed Hessian of the negative log posterior by central finite differences.

If this Hessian is positive definite,

```text
Cov(parameters | y) ~= H_obs^{-1}
```

and the Laplace log evidence is evaluated from the same independently
recomputed observed Hessian. The Hessian is never made positive definite by
eigenvalue clipping. Ill-conditioning is reported separately from validity.

## Deterministic and stochastic latent states

A fixed `initial_state` with `initial_state_covariance=None` and
`process_covariance=None` defines deterministic state evolution conditional on
static parameters and observations.

For a stochastic state model,

```text
x_{t+1} = f(x_t, theta, u_t, y_t) + eta_t
eta_t ~ N(0, Q_t)
```

`Q_t` is the process covariance and has dimension equal to the latent-state
dimension, not the number of evolution parameters.

For Gaussian outcomes,

```text
y_t = g(x_t, phi, u_t) + epsilon_t
epsilon_t ~ N(0, R_t)
```

where `R_t` is the observation covariance. Bernoulli and categorical outcomes
do not use an additional Gaussian observation covariance.

## Filtering likelihood

When latent states are uncertain, the subject likelihood is the predictive
state-space likelihood

```text
p(y_t | y_1:t-1, theta, phi)
 = integral p(y_t | x_t, phi)
            p(x_t | y_1:t-1, theta, phi) dx_t.
```

This filtering likelihood is evaluated at every static parameter vector
proposed by the optimizer. `latent_uncertainty="filtered"` does not create this
likelihood; it only asks BayesGBM to retain and display the internal filtered
state covariance after fitting.

Gaussian observations use an extended Kalman update. Bernoulli and categorical
observations use a local Laplace-Gaussian update with Fisher curvature. The
discrete filter is approximate Bayesian filtering and is labelled accordingly.

## Latent uncertainty

BayesGBM exposes exactly two latent uncertainty summaries.

### Propagated

Static parameters are sampled from the joint Laplace posterior and the same
state model is rerun for every sample. This is the same algorithm for all
observation families. The output contains trial-wise covariance and empirical
credible intervals. For stochastic state models, the spread of conditional
filtered means isolates static-parameter-induced uncertainty.

### Filtered

The internal filtered covariance `P_t` is retained conditional on the MAP
static parameters. The full covariance is the numerical result; plot shadows
are marginal Gaussian summaries derived from its diagonal.

## Diagnostics

BayesGBM reports multi-start convergence, observed-Hessian geometry, posterior
covariance/correlation, prior-preconditioned likelihood curvature, and numerical
local identifiability. Local identifiability is a numerical sensitivity/rank
diagnostic and is not a proof of global structural identifiability.
