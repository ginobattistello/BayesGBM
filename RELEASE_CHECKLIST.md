# Release checklist

Before the first public/Zenodo release:

- [ ] Replace `BayesGBM contributors` in `pyproject.toml` and `CITATION.cff`
      with final author metadata.
- [ ] Confirm the final GitHub repository URL.
- [ ] Run `python -m pytest -q` in a clean Python >=3.10 environment.
- [ ] Run every script in `bayesgbm/dev/`.
- [ ] Run the eight tutorials in `examples/`.
- [ ] Confirm GitHub Actions passes all supported Python versions.
- [ ] Review `LICENSE` and `NOTICE.md` attribution.
- [ ] Tag `v0.1.0` only after the public API is frozen for that release.
- [ ] Connect the GitHub repository to Zenodo and archive the tagged release.
- [ ] Add the Zenodo DOI back to `CITATION.cff` and README after minting.
