def multiple_linear_regression(X, y, n_draws=1000):
    """Perform a Bayesian multiple linear regression.
    
    Args:
        X (dataframe): Predictor variables are in different columns.
        y (ndarray/Series): The outcome variable.
        
    Returns:
    
    """
    
    # Standardize both predictor and outcome variables.
    if (is_standardized(X)) & (is_standardized(y)):
        pass
    else:
        X, _, _ = standardize(X)
        y, _, _ = standardize(y)
    
    # For explanation of how predictor variables are handled, see:
    # https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html
    with pm.Model(coords={"predictors": X.columns.values}) as model:
        # Define priors
        beta0 = pm.Normal("beta0", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2, dims="predictors")

        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Exponential("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))
        
        mu = beta0 + pm.math.dot(X, beta)
        sigma = pm.Uniform("sigma", 10**-5, 10)
        
        # Define likelihood
        likelihood = pm.StudentT("likelihood", nu=nu, mu=mu, lam=1/sigma**2, observed=y)
    
        # Sample the posterior
        idata = pm.sample(draws=n_draws)
        
        return model, idata