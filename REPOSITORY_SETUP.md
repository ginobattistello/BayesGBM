# Create the BayesGBM repository

1. Create an empty GitHub repository named `BayesGBM`.
2. Review `pyproject.toml` and `CITATION.cff`; repository URLs are set to
   `https://github.com/ginobattistello/BayesGBM`.
3. Replace the placeholder contributor metadata before the first Zenodo release.
4. From this directory run:

```bash
git init
git add .
git commit -m "Initial BayesGBM implementation"
git branch -M main
git remote add origin https://github.com/ginobattistello/BayesGBM.git
git push -u origin main
```

5. Validate in a clean environment:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

6. Run the scientific reference scripts in `bayesgbm/dev/`.
7. Enable GitHub Actions and require the test workflow before merging.
8. When the API is stable, create a tagged GitHub release and connect the
   repository to Zenodo for an archival DOI.

## Attribution

Keep `LICENSE`, `NOTICE.md`, and the Piray et al. (2019) citation. BayesGBM is
an independent software project with explicit CBM/HBI methodological lineage.
