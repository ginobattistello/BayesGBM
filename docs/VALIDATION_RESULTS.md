# Validation results for BayesGBM 0.1.0

Validation snapshot: 2026-09-11.

Environment used for the final local validation:

- Python 3.13.5;
- NumPy 2.3.5;
- SciPy 1.17.0.

Compatibility was additionally checked by parsing every Python source file
against Python 3.10 grammar. GitHub Actions is configured to run the automated
suite on Python 3.10, 3.11, 3.12 and 3.13.

## Automated suite

```text
38 passed
```

The suite covers priors/fixed parameters, family likelihoods, centralized
validation, exact linear-Gaussian/Kalman equivalence, Bernoulli/categorical
filter reference cases, MAP/Laplace fitting, propagated and filtered latent
uncertainty, display-after-`display=False`, diagnostics, predictive checks,
BMS, and hierarchical group refitting.

## Scientific reference scripts

All seven scripts in `bayesgbm/dev/` completed successfully.

Selected numerical results:

- linear-Gaussian filter vs exact Kalman recursion: maximum errors around
  `1e-13` for state means, variances and predictive log likelihood;
- Bernoulli one-step predictive probability vs dense quadrature: absolute
  approximation error about `1.2e-3` in the documented reference case;
- categorical one-step predictive probability vs quadrature: absolute
  approximation error about `4.1e-3` in the documented reference case;
- posterior-sampling variance propagation in the linear reference case:
  relative Monte-Carlo error about `0.55%` with 200,000 samples;
- deliberately redundant two-parameter observation model: numerical local
  identifiability rank `1/2`, as expected.

These checks validate implementation behavior in controlled reference cases;
they do not imply that the approximate discrete filter is exact for arbitrary
nonlinear/non-Gaussian models.
