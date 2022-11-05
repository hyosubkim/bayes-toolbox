def robust_linear_regression(x, y, n_draws=1000):
    """Perform a robust linear regression with one predictor.
    
    Args:
        x (ndarray): The standardized predictor (independent) variable.
        y (ndarray): The standardized outcome (dependent) variable.
        n_draws: Number of random samples to draw from the posterior.
        
    Returns: 
        PyMC Model and InferenceData objects.
    """
    
    # Make sure variables are standardized (within reasonable precision). 
    assert (is_standardized(x)) & (is_standardized(y)), f"Inputs must be standardized."

    with pm.Model() as model:
        # Define priors
        beta0 = pm.Normal('beta0', mu=0, sigma=2)
        beta1 = pm.Normal('beta1', mu=0, sigma=2)
        
        sigma = pm.Uniform('sigma', 10**-3, 10**3)
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))
        
        mu = beta0 + beta1 * x

        # Define likelihood 
        likelihood = pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, observed=y)
        
        # Sample from posterior
        idata = pm.sample(draws=n_draws)
    
    return model, idata
