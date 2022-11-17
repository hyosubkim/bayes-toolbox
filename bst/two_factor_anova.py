def two_factor_anova(x1, x2, y):
    
    mu_y = y.mean()
    sigma_y = y.std()

    x1 = df.Pos
    x2 = df.Org

    a_shape, a_rate = bst.gamma_shape_rate_from_mode_sd(sigma_y / 2 , 2 * sigma_y)
    x1_vals, levels1, n_levels1 = bst.parse_categorical(x1)
    x2_vals, levels2, n_levels2 = bst.parse_categorical(x2)

    with pm.Model(coords={"rank": levels1, "dept": levels2}) as model:

        #a0 = pm.Normal('a0', mu_y, tau=1/(sigma_y*5)**2)
        a0_tilde = pm.Normal('a0_tilde', mu=0, sigma=1)
        a0 = pm.Deterministic('a0', mu_y + sigma_y * 5 * a0_tilde)

        sigma_a1 = pm.Gamma('sigma_a1', a_shape, a_rate)
        #a1 = pm.Normal('a1', 0.0, tau=1/sigma_a1**2, shape=n_levels1)
        a1_tilde = pm.Normal('a1_tilde', mu=0, sigma=1, dims="rank")
        a1 = pm.Deterministic('a1', 0.0 + sigma_a1*a1_tilde)

        sigma_a2 = pm.Gamma('sigma_a2', a_shape, a_rate)
        #a2 = pm.Normal('a2', 0.0, tau=1/sigma_a2**2, shape=n_levels2)
        a2_tilde = pm.Normal('a2_tilde', mu=0, sigma=1, dims="dept")
        a2 = pm.Deterministic('a2', 0.0 + sigma_a2*a2_tilde)

        sigma_a1a2 = pm.Gamma('sigma_a1a2', a_shape, a_rate)
        #a1a2 = pm.Normal('a1a2', 0.0, 1/sigma_a1a2**2, shape=(n_levels1, n_levels2))
        a1a2_tilde = pm.Normal('a1a2_tilde', mu=0, sigma=1, dims=("rank", "dept"))
        a1a2 = pm.Deterministic('a1a2', 0.0 + sigma_a1a2*a1a2_tilde)

        mu = a0 + a1[x1_vals] + a2[x2_vals] +a1a2[x1_vals, x2_vals]
        sigma = pm.Uniform('sigma', sigma_y / 100, sigma_y * 10)

        likelihood = pm.Normal('likelihood', mu, sigma=sigma, observed=y) 

        idata = pm.sample(nuts={'target_accept': 0.95})