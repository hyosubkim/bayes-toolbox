def metric_outcome_one_nominal_one_metric_predictor(x, x_met, y, mu_x_met, mu_y, sigma_x_met, sigma_y, n_draws=1000):
    """
    
    """
    x_vals, levels, n_levels = parse_categorical(x)
    
    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)
    
    with pm.Model(coords={"groups": levels}) as model:
        # 'a' indicates coefficients not yet obeying sum-to-zero contraint
        sigma_a = pm.Gamma('sigma_a', alpha=a_shape, beta=a_rate)
        a0 = pm.Normal('a0', mu=mu_y, sigma=sigma_y * 5)
        a = pm.Normal('a', 0.0, sigma=sigma_a, dims="groups")
        a_met = pm.Normal("a_met", mu=0, sigma=2 * sigma_y / sigma_x_met)
        # Note that in Warmhoven notebook he uses SD of residuals to set
        # lower bound on sigma_y
        sigma_y = pm.Uniform('sigma_y', sigma_y / 100, sigma_y * 10)
        mu = a0 + a[x_vals] + a_met * (x_met - mu_x_met)
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma_y, observed=y)

        # Convert a0, a to sum-to-zero b0, b 
        b0 = pm.Deterministic('b0', a0 + at.mean(a) + a_met * (-mu_x_met))
        b = pm.Deterministic('b', a - at.mean(a)) 
        
        idata = pm.sample(draws=n_draws)

        return model, idata