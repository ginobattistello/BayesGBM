# Development

BayesGBM requires Python >= 3.10.

Install for development:

```bash
python -m pip install -e ".[dev]"
```

Run the merge-gate tests:

```bash
python -m pytest -q
```

Then run the scientific reference scripts:

```bash
for f in bayesgbm/dev/*.py; do python "$f"; done
```

## Numerical rules

- MAP search: multi-start L-BFGS-B.
- Do not use L-BFGS-B's inverse-Hessian approximation for inference.
- Recompute the observed negative-log-posterior Hessian independently at the final MAP.
- Do not clip observed-Hessian eigenvalues.
- A finite MAP and a valid Laplace approximation are separate outcomes.
- Bernoulli/categorical state filtering is approximate and must remain explicitly labelled.
- Modeller programming errors should propagate; legitimate invalid parameter regions should be flagged by runtime sanity diagnostics.

## Validation philosophy

Every new numerical approximation needs both:

1. a readable scientific reference script in `bayesgbm/dev/`, and
2. an automated regression/reference test in `tests/`.

Prefer analytic references. When they are unavailable, use high-accuracy numerical integration or a simpler exact special case.
