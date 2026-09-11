import numpy as np
import pytest
from bayesgbm import Config
from bayesgbm.validation import validate_fit_spec


def test_rejects_nan_y(binary_model, binary_data):
    bad = [{"y": binary_data[0]["y"].astype(float).copy(), "u": binary_data[0]["u"]}]
    bad[0]["y"][2] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_fit_spec(bad, binary_model, Config())


def test_rejects_binary_codes(binary_model, binary_data):
    bad = [{"y": binary_data[0]["y"].copy(), "u": binary_data[0]["u"]}]
    bad[0]["y"][0] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_fit_spec(bad, binary_model, Config())


def test_rejects_mismatched_input_length(binary_model, binary_data):
    bad = [{"y": binary_data[0]["y"], "u": {"reward": np.zeros(5)}}]
    with pytest.raises(ValueError, match="expected 30"):
        validate_fit_spec(bad, binary_model, Config())


def test_filtered_uncertainty_rejected_for_deterministic_model(binary_model, binary_data):
    with pytest.raises(ValueError, match="requires initial-state uncertainty"):
        validate_fit_spec(binary_data, binary_model, Config(latent_uncertainty="filtered"))


def test_latent_sampling_settings_discarded_when_not_propagated():
    cfg = Config(latent_uncertainty="none", latent_samples=55, latent_interval=.8)
    assert cfg.latent_samples is None
    assert cfg.latent_interval is None
