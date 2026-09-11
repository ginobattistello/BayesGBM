# Validation strategy

BayesGBM uses two complementary validation layers.

## Automated tests

`tests/` is the merge gate. The suite checks prior normalization and fixed
parameters, family likelihood formulas, centralized preflight validation,
linear-Gaussian filtering against exact Kalman recursions, discrete filtering
against numerical integration reference cases, MAP/Laplace fitting, latent
uncertainty, diagnostics, predictive simulation, BMS, hierarchical refitting,
and plotting after `display=False`.

Run:

```bash
python -m pytest -q
```

## Scientific reference scripts

`bayesgbm/dev/` contains readable method checks. Each script states the
mathematical question, the reference calculation, and a failure criterion.
They are intentionally separate from the tutorials in `examples/`.

Current reference scripts cover:

1. elementary observation-family likelihoods;
2. observed-Hessian conditioning without curvature clipping;
3. exact linear-Gaussian/Kalman equivalence;
4. Bernoulli Laplace-Gaussian filtering versus quadrature;
5. categorical Laplace-Gaussian filtering versus quadrature;
6. posterior sampling versus analytic linear uncertainty propagation;
7. a deliberately rank-deficient local-identifiability example.

Run all scripts with:

```bash
for f in bayesgbm/dev/*.py; do python "$f"; done
```
