# CBM Python — revised full implementation TODO (v3)

**Target repository:** https://github.com/ginobattistello/cbm_python  
**Revision date:** 2026-09-10  
**Goal:** redesign the public model API around a single generative state-model specification, add two clearly separated latent-uncertainty outputs, centralize validation before optimization, and integrate the previously planned diagnostics and predictive checks.

---

## 0. Final design principles

1. The modeller defines the scientific model once.
2. Use one consistent notation:
   - `theta`: static evolution parameters
   - `phi`: static observation parameters
   - `x_t`: dynamic latent state
   - `u_t`: known trial input
   - `y_t`: observed outcome
3. Do not make `latent_uncertainty` change the scientific likelihood. The generative model determines the likelihood; `latent_uncertainty` controls which latent uncertainty is retained/reported.
4. Keep exactly two latent-uncertainty outputs:
   - `"propagated"`: uncertainty induced by the posterior over static parameters
   - `"filtered"`: internal uncertainty of latent states
5. Remove the outer optimization GN polish from the redesigned API. Use multi-start L-BFGS-B for MAP and a separate observed posterior Hessian for Laplace inference.
6. Use no arbitrary default process or Gaussian observation variance.
7. Centralize model/data/config validation before optimization.
8. Use `random_state=42` as the default root seed for every stochastic component and spawn independent child RNG streams.
9. Examples are tutorials: standalone, short, commented, and directly runnable.
10. Keep final observed-Hessian diagnostics strict: no eigenvalue clipping.

---

# PART I — GENERATIVE MODEL API

## 1. Data

Replace the model-input key `"X"` with `"u"`:

```python
data[n] = {
    "y": y,
    "u": u,
}
```

`u` is generic. It has no reserved fields such as `"reward"` or `"stimulus"`.

Recommended accepted forms:

```python
u = array_like_with_first_dimension_T
```

or

```python
u = {
    "any_name": trialwise_array,
    "another_name": trialwise_array,
    "constant": scalar,
}
```

For dictionary inputs, array-like values must have first dimension `T`; scalars are treated as constants. CBM creates `u_t` generically by slicing the trialwise values. Field names are entirely modeller-defined.

The observed outcome `y_t` is passed separately to the evolution function; it is never inserted into `u`.

---

## 2. State model

Create `cbm/state_model.py`.

```python
model = StateModel(
    evolution=evolution,
    observation=observation,
    family="bernoulli",
    priors=priors,
    initial_state=[0.5, 0.5],
    initial_state_covariance=None,
    process_covariance=None,
    observation_covariance=None,
    state_names=None,
    name=None,
)
```

Scientific functions:

```python
def evolution(x, theta, u_t, y_t):
    # Return x_{t+1}.
    ...

def observation(x, phi, u_t):
    # Return the observation prediction for trial t.
    ...
```

Trial order:

```text
x_t
 -> observation(x_t, phi, u_t)
 -> likelihood of y_t
 -> evolution(x_t, theta, u_t, y_t)
 -> x_{t+1}
```

This avoids duplicate likelihood/evolution implementations.

---

## 3. Observation families

`family` is one of:

```python
"gaussian"
"bernoulli"
"categorical"
```

Observation return contract:

| family | `observation(...)` returns |
|---|---|
| Gaussian | predicted mean, scalar or vector |
| Bernoulli | scalar `P(y=1)` |
| categorical | length-K probability vector |

CBM constructs the family likelihood internally.

For categorical outcomes, use one Python-native convention everywhere:

```text
0, 1, ..., K-1
```

Not every category needs to appear in every subject.

---

# PART II — PARAMETERS AND PRIORS

## 4. Static parameter priors

Create:

```python
GaussianPrior(
    mean=...,
    covariance=...,
    names=None,
)
```

and:

```python
Priors(
    evolution=GaussianPrior(...),
    observation=GaussianPrior(...),
)
```

Mathematically:

theta ~ N(mu_theta, Sigma_theta)

phi ~ N(mu_phi, Sigma_phi)

Internally concatenate `[theta, phi]`.

The model stores the slices needed to recover `theta` and `phi`.

Remove the need for the modeller to specify `Config.d`; infer the static parameter dimension from the prior blocks.

### Covariance convenience

For parameter priors:

- scalar -> isotropic covariance
- vector -> diagonal covariance
- matrix -> full covariance

A zero marginal variance fixes that parameter at its prior mean. For a full covariance, a fixed component must also have a zero row/column of cross-covariances.

Names are optional. If supplied, they must be unique strings with exactly one name per parameter.

---

## 5. Initial latent state

`initial_state` is the fixed starting mean/value of `x_0`.

```python
initial_state=[0.5, 0.5]
```

means the initial state is known exactly unless uncertainty is explicitly supplied.

```python
initial_state_covariance=None
```

means `P0 = 0`.

If supplied, `initial_state_covariance` follows the same input convenience as priors:

- scalar -> isotropic covariance
- vector -> diagonal covariance
- matrix -> full covariance

It must be symmetric positive semidefinite. Zero variances are valid.

The initial state is not automatically estimated as another static parameter in this implementation.

---

# PART III — STATE AND OBSERVATION NOISE

## 6. Process covariance Q

Use the precise name:

```python
process_covariance
```

rather than `state_noise`.

The stochastic state equation is:

```text
x_{t+1} = f(x_t, theta, u_t, y_t) + eta_t
eta_t ~ N(0, Q_t)
```

`Q_t` describes uncertainty in the evolution of the latent state, so its shape is `(n_x, n_x)`, where `n_x` is the number of latent-state dimensions. It has no direct relationship to the number of evolution parameters.

Accepted forms:

- `None` -> deterministic evolution, `Q_t = 0`
- scalar -> `q I`
- vector of length `n_x` -> diagonal covariance
- `(n_x,n_x)` matrix -> full covariance
- callable `process_covariance(theta, u_t)` -> trial-/parameter-dependent covariance

The callable form lets the modeller estimate process noise by placing transformed noise-scale parameters inside `theta`.

---

## 7. Observation covariance R

Use:

```python
observation_covariance
```

rather than `observation_noise`.

For Gaussian outcomes:

```text
y_t = g(x_t, phi, u_t) + epsilon_t
epsilon_t ~ N(0, R_t)
```

`R_t` represents residual/measurement variability in observed data after conditioning on the latent state.

It is required for Gaussian models because a predicted mean alone does not define a Gaussian likelihood.

Accepted forms:

- scalar
- diagonal vector
- full covariance matrix
- callable `observation_covariance(phi, u_t)`

If measurement variance is known scientifically, fix it. If unknown, estimate it as part of `phi`; for a scalar standard deviation, a recommended parameterization is:

```python
def observation_covariance(phi, u_t):
    sigma = np.exp(phi[sigma_index])
    return sigma**2
```

with a prior on `log(sigma)`. This guarantees positivity without adding a new prior family.

For Bernoulli and categorical outcomes, `observation_covariance` must be `None`; their observation distributions already determine the conditional variance.

---

# PART IV — WHEN FILTERING CHANGES THE LIKELIHOOD

## 8. Separate generative-model stochasticity from display configuration

Important correction:

`latent_uncertainty="filtered"` must **not itself change the likelihood**.

The likelihood is determined by the generative model.

A deterministic latent model has `P0=0` and `Q_t=0`. Then `x_t` is exactly determined by static parameters, inputs and previous outcomes, and CBM evaluates `log p(y_t | x_t, phi)`.

A stochastic/uncertain latent model has non-zero initial-state uncertainty and/or process covariance. Then the correct predictive likelihood is:

```text
p(y_t | y_1:t-1, theta, phi)
 = integral p(y_t | x_t, phi)
            p(x_t | y_1:t-1, theta, phi) dx_t
```

Therefore a filter is part of the likelihood evaluation.

### When is it passed to the optimizer?

At **every candidate static parameter vector** proposed by L-BFGS-B:

```text
candidate [theta, phi]
 -> run deterministic recursion OR filter through all trials
 -> obtain per-trial predictive log likelihoods
 -> sum likelihood
 -> add static parameter log prior
 -> return negative log posterior to L-BFGS-B
```

Filtering is therefore not a post-fit plotting operation when the scientific model contains latent-state uncertainty.

A required regression test is that when `P0=0` and `Q=0`, the stochastic/filtering machinery reduces to the deterministic likelihood.

---

# PART V — FILTERING

## 9. Gaussian prediction step

Maintain a Gaussian approximation:

```text
x_t | y_1:t-1 ~ N(m_t^-, P_t^-)
```

For nonlinear state evolution:

```text
m_{t+1}^- = f(m_t, theta, u_t, y_t)
P_{t+1}^- = F_t P_t F_t' + Q_t
```

where `F_t` is the Jacobian of the evolution function with respect to the latent state:

```text
F_t = d f(x, theta, u_t, y_t) / d x
```

evaluated at the current state estimate.

Its shape is `(n_x, n_x)`.

CBM computes `F_t` numerically by default. The modeller does not provide it.

---

## 10. Gaussian observations

For Gaussian observations CBM uses the standard extended-Kalman update for nonlinear `g`.

CBM numerically computes the observation Jacobian with respect to `x`.

For linear-Gaussian models this must reduce numerically to the exact Kalman filter.

---

## 11. Bernoulli and categorical observations

A Gaussian predictive state multiplied by a Bernoulli/categorical likelihood is no longer Gaussian.

Use a documented **Laplace-Gaussian filtering update with Fisher scoring**:

1. start from the Gaussian predictive state;
2. combine its log density with the Bernoulli/categorical log likelihood;
3. find the local posterior state mode;
4. use Fisher/expected likelihood curvature for a stable local Gaussian curvature;
5. approximate the filtered state by a Gaussian around that mode;
6. compute the corresponding approximate predictive log likelihood contribution.

This is a standard class of approximate inference for nonlinear/non-Gaussian state-space models. It is not exact Bayesian filtering and should never be described as such.

The public method label should be explicit, e.g. `"laplace_gaussian_fisher"`.

Do not call the discrete update an ordinary EKF.

The implementation should be validated against high-accuracy numerical quadrature in one-dimensional Bernoulli and categorical toy models.

---

# PART VI — OPTIMIZATION AND LAPLACE INFERENCE

## 12. Remove outer GN polish, but keep the Hessians strictly separated

For the redesigned API, remove the post-L-BFGS Gauss-Newton polish.

Use:

```text
multi-start L-BFGS-B
 -> final MAP
 -> independently recompute observed posterior Hessian at that MAP
 -> Laplace covariance/evidence
```

The critical distinction is:

```text
optimizer curvature != Laplace curvature
```

During MAP search, L-BFGS-B maintains its own quasi-Newton approximation internally. CBM must never use that approximation for posterior covariance or evidence.

After the final MAP is selected, CBM independently evaluates:

```text
H_obs = Hessian[-log p(y, theta, phi)] at the final MAP
```

using central finite differences by default. `H_obs` is the observed posterior curvature of the final objective and is the Hessian required by the Laplace approximation:

```text
posterior covariance ~= inverse(H_obs)
```

and the Laplace evidence uses the same independently recomputed `H_obs`.

This preserves the numerical separation that motivated the previous refactor. Removing the outer GN polish does **not** mean reusing an optimization Hessian; it removes an extra optimization heuristic entirely.

For filtered models, `H_obs` is computed by finite-differencing the complete filtered log-posterior objective. The local Fisher curvature used *inside* a Bernoulli/categorical filtering update must never be reused as `H_obs` and must never enter the Laplace evidence directly.

Reasons for removing the outer GN polish:

- its score-outer-product curvature is only an optimization approximation;
- it is not the observed posterior Hessian used by Laplace;
- filtered likelihoods can already contain local approximation machinery;
- nested numerical curvature makes another polish layer harder to interpret;
- multi-start L-BFGS-B provides the MAP search;
- removing it simplifies code, diagnostics and documentation.

The Fisher-scoring step inside the discrete **filter** remains. It is a different algorithmic object and must not be called the optimizer GN polish.

Keep the independent central finite-difference observed Hessian as default, optional autodiff only where the entire final objective supports it, no eigenvalue clipping, and the current PD/condition-number/Laplace-valid diagnostics.

---

# PART VII — LATENT UNCERTAINTY OUTPUTS

## 13. Configuration

Exactly:

```python
latent_uncertainty = "none" | "propagated" | "filtered"
```

Default:

```python
latent_uncertainty="none"
latent_samples=1000
latent_interval=0.95
```

`latent_samples` and `latent_interval` are meaningful **only** for `"propagated"`.

During configuration resolution:

```text
if latent_uncertainty != "propagated":
    resolved latent_samples = None
    resolved latent_interval = None
```

They have no effect and are omitted from downstream calculations/metadata. If the modeller explicitly sets non-default values while using another mode, issue one clear warning that they were ignored.

---

## 14. Propagated static-parameter uncertainty

Use one algorithm for Gaussian, Bernoulli and categorical outcomes.

Sample from the full static Laplace posterior:

```text
[theta, phi]^(s) ~ N([theta_hat, phi_hat], Sigma_theta_phi)
```

Fixed parameters remain fixed.

For a deterministic state model, rerun the state recursion for every sample.

For a stochastic state model, rerun the same filter for every static-parameter sample and retain the conditional filtered state mean. The spread between these means is the **static-parameter-induced** component only; the internal state covariance is not added.

For each latent variable/state retain:

- mean
- trial-wise covariance
- marginal SD
- empirical posterior quantile bounds
- interval mass
- `uncertainty_type="propagated"`
- method label

Empirical quantiles are retained because nonlinear propagation can make marginal state uncertainty asymmetric.

If the final static Laplace posterior is invalid, retain the MAP latent trajectory but do not manufacture propagated uncertainty.

---

## 15. Filtered internal uncertainty

`latent_uncertainty="filtered"` stores the filter's `m_t` and `P_t`.

This is uncertainty in the latent state conditional on the fitted static parameters.

It is valid only for a model with actual latent-state uncertainty (non-zero `P0` and/or a process covariance specification). If both are exactly zero, preflight should raise a clear error and suggest `"propagated"` or `"none"`.

The primary uncertainty object is `P_t`. The display may derive a 95% Gaussian marginal band from its diagonal, but these bounds are display summaries, not separately estimated intervals.

---

# PART VIII — STANDARD LATENT OUTPUT

## 16. One schema

Use one schema for every latent/state trajectory:

```python
{
    "mean": ...,
    "covariance": ...,
    "sd": ...,
    "interval_low": ...,
    "interval_high": ...,
    "interval_mass": ...,
    "interval_method": ...,
    "uncertainty_type": "none" | "propagated" | "filtered",
    "method": ...,
}
```

For propagated uncertainty, empirical interval fields are populated.

For filtered Gaussian state uncertainty, covariance is primary and explicit interval fields remain `None`.

---

# PART IX — CENTRALIZED PREFLIGHT VALIDATION

## 17. Create one validation layer

Create `cbm/validation.py` with one entry point:

```python
validated = validate_fit_spec(data, model, config)
```

Call it once before optimization from `individual_fit`, and reuse it from HBI entry points.

It should both normalize acceptable shorthand and validate the resolved model specification.

Most static checks should move here. Runtime checks remain inside the objective only for problems that can arise at candidate parameter values.

### Required preflight checks

- data is a non-empty subject sequence;
- every subject contains `y` and `u`;
- `y` is non-empty and contains no NaN/Inf;
- numeric trialwise entries of `u` contain no NaN/Inf;
- trialwise `u` lengths match `len(y)`;
- subject input schemas are compatible;
- Bernoulli `y` contains only `0/1`;
- categorical `y` contains integer labels only;
- categorical labels are in `0,...,K-1` after probing `observation`;
- Gaussian outcomes are finite numeric arrays with dimension matching the observation prediction;
- prior means are finite one-dimensional arrays;
- scalar/vector/matrix prior covariance expands to the correct parameter dimension;
- prior covariance is finite, symmetric and positive semidefinite;
- zero-variance fixed parameters have zero cross-covariance rows/columns;
- evolution/observation parameter counts match their prior blocks;
- optional parameter names have exactly the correct lengths;
- names are strings and unique within the complete static parameter space;
- `state_names`, if supplied, match `initial_state`;
- `initial_state` is finite and one-dimensional;
- initial covariance shorthand expands to `(n_x,n_x)` and is symmetric PSD;
- `process_covariance`, when fixed, expands to `(n_x,n_x)` and is symmetric PSD;
- callable process covariance returns a finite symmetric PSD matrix of the correct shape at the prior mean;
- Gaussian `observation_covariance` is supplied and has correct positive-definite dimensions;
- Bernoulli/categorical models reject a non-`None` observation covariance;
- `evolution` returns a finite state with exactly the same shape as its input state;
- `observation` returns finite values;
- Bernoulli predictions lie in `[0,1]`;
- categorical probabilities are non-negative, finite, have `K>=2`, and sum to one within tolerance;
- model functions are deterministic at fixed arguments: evaluate representative calls twice and compare;
- numerical state Jacobians needed for filtering have correct shape and finite values;
- bounds/inits match the inferred number of free static parameters;
- fixed parameter values are compatible with bounds;
- Hessian settings are valid;
- `latent_uncertainty` is one of the allowed values;
- filtered uncertainty is not requested for a completely deterministic state model;
- `latent_samples>=2` and `0<latent_interval<1` only when propagated uncertainty is active;
- `random_state` is an integer or `None`;
- a complete model evaluation at prior means returns finite per-trial log likelihoods for every subject.

### Runtime sanity checks that must remain

Preflight cannot guarantee valid behavior for every parameter candidate. Add one centralized runtime helper, for example:

```python
safe_objective_evaluation(...)
```

with a compact `RuntimeSanityTracker`.

At every objective evaluation it should check:

- finite candidate static parameters;
- finite state/evolution outputs;
- valid Bernoulli/categorical probabilities;
- finite Gaussian predictions;
- valid callable `Q_t` and `R_t`;
- finite predictive covariance matrices;
- successful/finite filtering updates;
- finite per-trial and total log likelihood;
- finite log prior and log posterior.

Known numerical-domain failures should produce an invalid objective value and increment categorized counters, e.g.:

```text
nonfinite_state
invalid_probability
invalid_process_covariance
invalid_observation_covariance
filter_update_failure
nonfinite_loglik
```

Those counters are exposed in fit diagnostics.

Do **not** silently swallow arbitrary programming errors. Unexpected exceptions from modeller code should normally be re-raised with subject/trial/model context so bugs are distinguishable from legitimate invalid parameter regions.

---

# PART X — REPRODUCIBILITY

## 18. Default seed

Change the default to:

```python
random_state=42
```

for stochastic public APIs.

Keep the implementation simple: initialize one NumPy generator once per top-level operation,

```python
rng = np.random.default_rng(random_state)
```

and pass/use that generator sequentially for subjects, initializations, posterior sampling and simulations.

Do not repeatedly create a new `default_rng(42)` inside inner loops. A more elaborate `SeedSequence` hierarchy is not required for the first implementation. If parallel execution is added later, independent child streams can be introduced then.

---

# PART XI — DIAGNOSTICS AND MODEL CHECKS

## 19. Convergence diagnostics

Use information produced by multi-start L-BFGS-B.

Report number of starts, objective agreement, winning optimizer status/message, gradient norm, invalid objective evaluations, and hard-bound flags.

If compact endpoint parameters from each start are retained, additionally report final log-joint spread, maximum distance between endpoint parameter vectors, and whether objective and parameters agree.

Never compress this into an undocumented scalar convergence score.

---

## 20. Posterior/Hessian diagnostics

For free static parameters retain:

- full observed Hessian;
- posterior covariance;
- posterior marginal SD;
- posterior correlation;
- Hessian eigenvalues/eigenvectors;
- condition number;
- positive-definite flag;
- `laplace_valid`;
- `laplace_fragile`.

Reporting should split labels into evolution (`theta`) and observation (`phi`) blocks while preserving the full joint covariance.

---

## 21. Prior-preconditioned information

For Gaussian static priors:

```text
H_lik = H_post - Sigma0^{-1}
H_tilde = Sigma0^{1/2} H_lik Sigma0^{1/2}
```

Return eigenvalues, eigenvectors mapped to parameter names, negative-curvature flag, and directions where local likelihood curvature exceeds prior curvature.

Do not clip negative observed-likelihood curvature.

---

## 22. Numerical local identifiability

Compute the Jacobian of model observable predictions with respect to free static parameters.

For Gaussian/Bernoulli use the predicted mean/probability trajectory.

For categorical use a non-redundant `K-1` representation per trial to avoid simplex redundancy.

Use SVD to return singular values, numerical rank, nullity, condition number, and weak parameter combinations.

Call this **numerical local identifiability**, not proof of global structural identifiability.

---

## 23. Prior sensitivity

Refit the same observed dataset under user-specified plausible `Priors` objects.

Compare prior means/covariances with posterior static-parameter means and marginal variances, with evolution and observation blocks reported separately.

Do not mix BMS/evidence into this diagnostic.

---

## 24. Prior predictive checks

Use `StateModel` itself as the simulator.

Sample static parameters from priors, initialize the state, sample state innovations if `Q_t` is non-zero, and generate outcomes from the observation family. Gaussian outcomes use `R_t`.

No separate simulator class is required for the standard StateModel path.

---

## 25. Posterior predictive checks

Sample static parameters from the Laplace posterior and generate replicated datasets from the same StateModel.

For stochastic state models, include process noise and Gaussian observation noise.

Use the same simulation engine as prior predictive checks.

Do not confuse posterior predictive simulation with propagated latent uncertainty.

---

# PART XII — DISPLAY

## 26. Display semantics

`Config(display)` controls automatic showing only.

A fit produced with `Config(display=False)` must still support `fit.plot(subject=0)`.

Store compact realized arrays needed for plotting regardless of automatic display.

Compute requested latent uncertainty before any automatic figure is drawn.

---

## 27. Latent panel

Panel E:

- `"none"` -> MAP/conditional latent trajectory only;
- `"propagated"` -> posterior mean + empirical quantile shadow;
- `"filtered"` -> filtered mean + display-derived Gaussian shadow from `P_t`.

Use the full covariance in numerical output; the shadow uses only marginal uncertainty.

---

## 28. Multi-line style

Centralize `_trace_styles(n)` with ordering:

```text
black solid
black dashed
dark gray solid
dark gray dashed
progressively lighter gray solid
progressively lighter gray dashed
...
```

Use this consistently in all multi-line trajectory panels.

---

# PART XIII — TESTING AND VALIDATION

## 29. Two-layer validation strategy

Use both scientifically grounded development scripts and pytest.

`cbm/dev/` contains human-readable proof-of-concept / method-validation scripts. Each script should state:

- the mathematical question being tested;
- the analytic or high-accuracy numerical reference;
- the expected result;
- the approximation being assessed;
- what would count as failure;
- one clear figure or compact numerical table where useful.

Recommended development scripts:

```text
01_likelihood_reference_cases.py
    direct Gaussian/Bernoulli/categorical likelihood checks

02_hessian_failure_cases.py
    observed-Hessian conditioning and Laplace fragility

03_linear_gaussian_filter.py
    nonlinear-filter code reduced to exact Kalman reference

04_bernoulli_filter_quadrature.py
    Laplace-Gaussian/Fisher update versus dense 1-D quadrature

05_categorical_filter_quadrature.py
    categorical update versus dense 1-D quadrature

06_propagated_uncertainty.py
    posterior sampling versus analytic linear covariance propagation

07_information_identifiability.py
    prior-preconditioned curvature and deliberately rank-deficient examples
```

These scripts are explanatory scientific validation artifacts, not the merge gate.

`tests/` contains all automated correctness requirements and is the merge gate.

Required automated references include:

| Component | Reference |
|---|---|
| Gaussian/Bernoulli/categorical likelihood | direct analytic formulas |
| deterministic recursion | hand-computed trajectory |
| prior covariance normalization | known scalar/diagonal/full cases |
| fixed parameters | exact fixed values / reduced dimension |
| initial covariance normalization | scalar/vector/full PSD cases |
| propagated uncertainty, linear model | analytic covariance transformation |
| propagated uncertainty, nonlinear model | seeded Monte Carlo invariants + quantiles |
| Gaussian filter | exact linear Kalman recursion and marginal likelihood |
| Bernoulli filter | high-accuracy 1D numerical quadrature |
| categorical filter | high-accuracy 1D numerical quadrature |
| deterministic filtering limit | `P0=Q=0` recovers deterministic likelihood |
| filtering covariance | symmetry/PSD/finite checks |
| preflight | one test per major invalid input class |
| reproducibility | same root seed gives identical results |
| RNG independence | spawned streams differ across subject/task |
| invalid/fragile Laplace | existing behavior retained |
| local identifiability | analytically rank-deficient toy model |
| prior-preconditioned spectrum | analytic quadratic model |
| prior sensitivity | controlled Gaussian toy fit |
| prior/posterior predictive | support, shape, reproducibility |
| display=False then plot | figure builds successfully |
| latent shadow | correct interval source is used |
| pickle round-trip | model/result survives serialization |
| HBI | StateModel likelihood works in hierarchical refits |
| BMS | existing log-evidence consumption unchanged |

Run the complete existing test suite after each implementation phase.

---

# PART XIV — TUTORIAL EXAMPLES

## 30. Example policy

Every example must run standalone, use the default seed 42, explain the scientific model first, explain `theta`, `phi`, and `x`, comment why each prior/noise choice exists, print a small interpretable result, and show the relevant plot.

Recommended sequence:

```text
01_individual_fit.py
    binary deterministic learning model
    latent_uncertainty="propagated"

02_continuous_fit.py
    continuous model with explicit observation covariance

03_bms.py

04_hbi.py

05_fixed_parameters.py

06_diagnostics.py
    convergence + Hessian + information + local identifiability

07_predictive_checks.py
    prior and posterior predictive checks

08_filtered_state_model.py
    one simple stochastic latent-state example
    latent_uncertainty="filtered"
```

Examples are tutorials, not API stress tests.

---

# PART XV — SOFTWARE IDENTITY / PUBLICATION

## 31. Publication strategy

The planned software is now substantially more than a numerical patch to the original CBM Python interface. It introduces a new generative-model API, filtering, propagated latent uncertainty, centralized validation, predictive checks, expanded diagnostics, and revised numerical/plotting behavior.

A separate citable software release is scientifically reasonable.

Provenance must remain explicit:

- retain the MIT license and original copyright notice for inherited code;
- state clearly that the software is derived from / inspired by CBM;
- cite Piray et al. (2019) for the CBM/HBI framework and the historical MAP/Laplace/HBI basis;
- distinguish inherited methodology from new algorithms and implementation;
- add `CITATION.cff`;
- archive tagged releases with Zenodo.

A Zenodo DOI provides a stable citable software archive; it is not peer review.

Once the redesign is stable, documented, tested, and has a clear research use case, a peer-reviewed research-software paper such as JOSS can be considered separately.

---

## 32. Working name for the redesigned toolbox

Use a working name during development rather than renaming the Python package immediately.

Preferred working title:

```text
GenBehav
```

Expanded description:

```text
GenBehav: a Python toolbox for generative behavioral modelling,
Bayesian parameter estimation, latent-state inference and model comparison
```

Reasons:

- it emphasizes generative modelling rather than only model comparison;
- it remains clearly oriented toward behavioral/computational modelling;
- it is short enough for a repository/package name (`genbehav`);
- it does not imply that the software implements only filtering or only hierarchical inference.

Avoid names such as `GenMod`, `PyGEM`, or `GBMT`: those names/acronyms are already used by unrelated scientific software/packages.

Before the first independent Zenodo release, perform a final availability check across GitHub, PyPI, Zenodo and general scholarly search.

Do not rename the import package during the core mathematical refactor. First stabilize the API/tests/documentation; then rename the project/package once, close to the independent release.

Suggested provenance statement for the future README/paper:

```text
GenBehav builds on the conceptual and methodological foundation of the
Computational Behavioral Modeling (CBM) framework, particularly its
individual MAP/Laplace estimation and hierarchical Bayesian inference,
while introducing a redesigned generative-model API, independent numerical
implementation, latent-state uncertainty methods, validation, diagnostics,
and predictive-checking workflow.
```

---

# References

- Current fork: https://github.com/ginobattistello/cbm_python
- Original CBM Python: https://github.com/payampiray/cbm_python
- VBA model structure: https://mbb-team.github.io/VBA-toolbox/wiki/Structure-of-VBA%27s-generative-model/
- VBA priors/inversion: https://mbb-team.github.io/VBA-toolbox/wiki/VBA-model-inversion-in-4-steps/
- Koyama et al., Approximate Methods for State-Space Models, JASA, DOI: 10.1198/jasa.2009.tm08326
- Zenodo GitHub integration: https://help.zenodo.org/docs/github/
