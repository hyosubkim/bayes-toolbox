def metric_outcome_one_nominal_predictor(x, y, mu_y, sigma_y, n_draws=1000):
    """
    
    """
    x_vals, levels, n_levels = parse_categorical(x)
    
    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)
    
    with pm.Model(coords={"groups": levels}) as model:
        # 'a' indicates coefficients not yet obeying sum-to-zero contraint
        sigma_a = pm.Gamma('sigma_a', alpha=a_shape, beta=a_rate)
        a0 = pm.Normal('a0', mu=mu_y, sigma=sigma_y * 5)
        a = pm.Normal('a', 0.0, sigma=sigma_a, dims="groups")

        sigma_y = pm.Uniform('sigma_y', sigma_y / 100, sigma_y * 10)
        likelihood = pm.Normal('likelihood', a0 + a[x_vals], sigma=sigma_y, observed=y)

        # Convert a0, a to sum-to-zero b0, b 
        m = pm.Deterministic('m', a0 + a)
        b0 = pm.Deterministic('b0', at.mean(m))
        b = pm.Deterministic('b', m - b0) 
        
        idata = pm.sample(draws=n_draws)

        return model, idata