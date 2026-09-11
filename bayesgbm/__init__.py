"""BayesGBM: Bayesian Generative Brain/Behavior Modelling."""

from .priors import GaussianPrior, Priors
from .state_model import StateModel
from .optimization import Config
from .individual_fit import individual_fit, FitResult
from .diagnostics import (
    convergence_diagnostics,
    posterior_hessian_diagnostics,
    prior_preconditioned_information,
    numerical_local_identifiability,
)
from .predictive import prior_predictive, posterior_predictive, simulate_subject
from .sensitivity import prior_sensitivity
from .model_selection import bms, BMSResult
from .hbi import hbi_main, HBIConfig, HBIResult

__all__ = [
    "GaussianPrior",
    "Priors",
    "StateModel",
    "Config",
    "individual_fit",
    "FitResult",
    "convergence_diagnostics",
    "posterior_hessian_diagnostics",
    "prior_preconditioned_information",
    "numerical_local_identifiability",
    "prior_predictive",
    "posterior_predictive",
    "simulate_subject",
    "prior_sensitivity",
    "bms",
    "BMSResult",
    "hbi_main",
    "HBIConfig",
    "HBIResult",
]

__version__ = "0.1.0"
