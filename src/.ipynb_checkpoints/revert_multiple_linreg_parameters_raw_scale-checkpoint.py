def revert_multiple_linreg_parameters_raw_scale(zeta0, zeta, mu_X, mu_y, sigma, sigma_X, sigma_y):
    
    beta0 = zeta0 * sigma_y + mu_y - np.sum(zeta * mu_X / sigma_X, axis=1) * sigma_y
    beta = (zeta / sigma_X) * sigma_y
    sigma = (sigma * sigma_y)
    
    return beta0, beta, sigma