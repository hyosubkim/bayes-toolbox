def BEST_paired(y1, y2=None, n_samples=1000):
    """BEST procedure on single sample or paired samples. 
    
    Args: 
        y1 (ndarray/Series): Either single sample or difference scores. 
        y2 (ndarray/Series): (Optional) If provided, represents the paired 
          sample (i.e., y2 elements are in same order as y1).
    Returns: 
        PyMC Model and InferenceData objects.
    """
    
    # Check to see if block variable was passed. If so, then this means the
    # goal is to compare difference scores on a within subjects variable 
    # (e.g., block). Otherwise, we are comparing location parameter to zero.
    if y2 is None:
        pass
    else:
        assert len(y1) == len(y2), f"There must be equal numbers of observations."
        # Convert pre and post to difference scores.
        y = y1 - y2
    
    # Calculate pooled empirical mean and SD of data to scale hyperparameters
    mu_y = y.mean()
    sigma_y = y.std()
                                                                     
    with pm.Model() as model:
        # Define priors
        mu = pm.Normal('mu', mu=mu_y, sd=sigma_y * 10)
        sigma = pm.Uniform('sigma', sigma_y / 10, sigma_y * 10)
        nu_minus1 = pm.Exponential('nu_minus_one', 1 / 29)
        nu = pm.Deterministic('nu', nu_minus1 + 1)
        
        # Define likelihood
        likelihood = pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, observed=y)

        # Sample from posterior
        idata = pm.sample(draws=n_draws)
        
    return model, idata
