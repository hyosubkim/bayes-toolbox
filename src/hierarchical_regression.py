def hierarchical_regression(x, y, subj):  
    """
    
    """
    
    # Convert subject variable to categorical dtype if it is not already
    subj_idx, subj_levels, n_subj = parse_categorical(subj)
    
    with pm.Model() as model:
        # Hyperpriors
        beta0 = pm.Normal('beta0', mu=0, tau=1/10**2)
        beta1 = pm.Normal('beta1', mu=0, tau=1/10**2)
        sigma0 = pm.Uniform('sigma0', 10**-3, 10**3)
        sigma1 = pm.Uniform('sigma1', 10**-3, 10**3)

        # The intuitive parameterization results in a lot of divergences.
        #beta0_s = pm.Normal('beta0_s', mu=beta0, sd=sigma0, shape=n_subj)
        #beta1_s = pm.Normal('beta1_s', mu=beta1, sd=sigma1, shape=n_subj)
        
        # See: http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/
        # for rationale of following reparameterization
        beta0_s_offset = pm.Normal('beta0_s_offset', mu=0, sd=1, shape=n_subj)
        beta0_s = pm.Deterministic('beta0_s', beta0 + beta0_s_offset * sigma0)

        beta1_s_offset = pm.Normal('beta1_s_offset', mu=0, sd=1, shape=n_subj)
        beta1_s = pm.Deterministic('beta1_s', beta1 + beta1_s_offset * sigma1)

        mu =  beta0_s[subj_idx] + beta1_s[subj_idx] * zx3

        sigma = pm.Uniform('sigma', 10**-3, 10**3)
        nu = pm.Exponential('nu', 1/29.)

        likelihood = pm.StudentT('likelihood', nu, mu=mu, sd=sigma, observed=zy3)  
        
        # Sample from posterior
        idata = pm.sample(draws=n_draws)
        
        return model, idata