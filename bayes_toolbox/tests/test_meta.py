import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from bayes_toolbox.meta import meta_binary_outcome, meta_normal_outcome_beta_version


def test_meta_binary_outcome():
    # Create test data
    z_t_obs = np.array([10, 20, 30])  # Example data
    n_t_obs = np.array([50, 60, 70])  # Example data
    z_c_obs = np.array([5, 15, 25])  # Example data
    n_c_obs = np.array([45, 55, 65])  # Example data
    study = ["Study1", "Study2", "Study3"]  # Example study names

    # Call the function
    model, idata = meta_binary_outcome(
        z_t_obs, n_t_obs, z_c_obs, n_c_obs, study, n_draws=1000
    )

    # Add assertions here to check the validity of the results
    assert isinstance(model, pm.Model)
    assert isinstance(idata, az.InferenceData)


def test_meta_normal_outcome_beta_version():
    # Create test data
    eff_size = np.array([0.5, 0.8, 1.2])  # Example data
    se_eff_size = np.array([0.1, 0.15, 0.2])  # Example data
    study = ["Study1", "Study2", "Study3"]  # Example study names

    # Call the function
    model, idata = meta_normal_outcome_beta_version(
        eff_size, se_eff_size, study, n_draws=1000
    )

    # Add assertions here to check the validity of the results
    assert isinstance(model, pm.Model)
    assert isinstance(idata, az.InferenceData)
