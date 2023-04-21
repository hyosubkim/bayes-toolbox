import numpy as np
from bayes_toolbox.glm import standardize

def test_mus():
    X = np.random.normal(loc=10, scale=20, size=(10,2))
    X_s, mu_X, _ = standardize(X)
    assert np.allclose(X_s.mean(axis=0), np.zeros_like(mu_X), atol=1e-4)
    
def test_sigmas():
    X = np.random.normal(loc=10, scale=20, size=(10,2))
    X_s, _, sigma_X = standardize(X)
    assert np.allclose(X_s.std(axis=0), np.ones_like(sigma_X), atol=1e-4)
    
    