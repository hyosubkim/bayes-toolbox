import arviz as az
import numpy as np
import pytest
import pymc as pm
from bayes_toolbox.glm import standardize, BEST_paired

def test_mus():
    X = np.random.normal(loc=10, scale=20, size=(10,2))
    X_s, mu_X, _ = standardize(X)
    assert np.allclose(X_s.mean(axis=0), np.zeros_like(mu_X), atol=1e-4)
    
def test_sigmas():
    X = np.random.normal(loc=10, scale=20, size=(10,2))
    X_s, _, sigma_X = standardize(X)
    assert np.allclose(X_s.std(axis=0), np.ones_like(sigma_X), atol=1e-4)s

# Define a test for BEST_paired
def test_BEST_paired():
    # Generate test data
    np.random.seed(0)
    y1 = np.random.normal(0, 1, 100)
    y2 = np.random.normal(0.1, 1, 100)

    # Call the BEST_paired function with test data
    model, idata = BEST_paired(y1, y2)

    # Perform assertions to check if the function output is as expected
    assert isinstance(model, pm.Model)
    assert isinstance(idata, az.data.inference_data.InferenceData)
    assert "mu" in model.named_vars
    assert "sigma" in model.named_vars
    assert "nu_minus_one" in model.named_vars
    assert "nu" in model.named_vars
    assert "likelihood" in model.named_vars
    assert "effect_size" in model.named_vars



