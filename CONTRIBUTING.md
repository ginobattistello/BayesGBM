# Contributing

Contributions are welcome after the initial API is stabilized.

Before opening a pull request:

1. run `python -m pytest -q`;
2. run relevant scripts in `bayesgbm/dev/`;
3. add tests for numerical or API changes;
4. document the scientific interpretation of new model assumptions;
5. keep examples short, standalone, and tutorial-oriented.

For changes to likelihoods, Hessians, filtering, evidence, BMS, or HBI, include a mathematical reference case rather than relying only on code-to-code comparison.
