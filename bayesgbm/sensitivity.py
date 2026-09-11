"""Prior-sensitivity refitting utilities."""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .individual_fit import individual_fit


@dataclass
class PriorSensitivityResult:
    prior_means: list[np.ndarray]
    prior_covariances: list[np.ndarray]
    posterior_means: list[np.ndarray]
    posterior_variances: list[np.ndarray]


def prior_sensitivity(data, model, priors_list, *, config):
    """Refit one model under explicit plausible prior specifications."""
    pmeans, pcovs, postmeans, postvars = [], [], [], []
    for priors in priors_list:
        m = replace(model, priors=priors)
        fit = individual_fit(data, m, config=config)
        pmeans.append(priors.mean.copy())
        pcovs.append(priors.covariance.copy())
        postmeans.append(fit.output.parameters.copy())
        vv = []
        for cov in fit.math.covariance:
            vv.append(np.full(m.n_parameters, np.nan) if cov is None else np.diag(cov))
        postvars.append(np.asarray(vv))
    return PriorSensitivityResult(pmeans, pcovs, postmeans, postvars)
