def BEST(y, group, n_draws=1000):
    """Implementation of John Kruschke's BEST test.
    
    Compares outcomes from two groups and estimates parameters.
    
    Args:
        y (ndarray/Series): The metric outcome variable.
        group: The grouping variable providing that indexes into y.
        n_draws: Number of random samples to draw from the posterior.
    
    Returns: 
        PyMC Model and InferenceData objects.
    """
    
    # Convert grouping variable to categorical dtype if it is not already
    if pd.api.types.is_categorical_dtype(group):
        pass
    else:
        group = group.astype('category')
        
    # Extract group levels and make sure there are only two
    level = group.cat.categories
    assert len(level) == 2, f"Expected two groups but got {len(level)}."
    
    # Split observations by group
    y_group1 = y[group==level[0]]
    y_group2 = y[group==level[1]]
    
    # Calculate pooled empirical mean and SD of data to scale hyperparameters
    mu_y = y.mean()
    sigma_y = y.std()
    
    # Arbitrarily set hyperparameters to the pooled empirical mean of data and 
    # twice pooled empirical SD, which applies very diffuse info to these 
    # quantities and does not favor one or the other a priori
    mu_m = mu_y
    mu_s = sigma_y * 2
                                                                     
    with pm.Model() as model:
        # Define priors
        group1_mean = pm.Normal("group1_mean", mu=mu_m, sigma=mu_s)
        group2_mean = pm.Normal("group2_mean", mu=mu_m, sigma=mu_s)
        group1_std = pm.Uniform("group1_std", lower=sigma_y / 10, upper=sigma_y * 10)
        group2_std = pm.Uniform("group2_std", lower=sigma_y / 10, upper=sigma_y * 10)
        
        # See Kruschke Ch 16.2.1 for in-depth rationale for prior on nu. The addition of 1 is to shift the
        # distribution so that the range of possible values of nu are 1 to infinity (with mean of 30).
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))
        
        # Define likelihood
        likelihood1 = pm.StudentT("group1", nu=nu, mu=group1_mean, sigma=group1_std, observed=y_group1)
        likelihood2 = pm.StudentT("group2", nu=nu, mu=group2_mean, sigma=group2_std, observed=y_group2)
        
        # Contrasts of interest
        diff_of_means = pm.Deterministic("difference of means", group1_mean - group2_mean)
        diff_of_stds = pm.Deterministic("difference of stds", group1_std - group2_std)
        effect_size = pm.Deterministic(
            "effect size", diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2)
        )
        
        # Sample from posterior
        idata = pm.sample(draws=n_draws)
        
    return model, idata