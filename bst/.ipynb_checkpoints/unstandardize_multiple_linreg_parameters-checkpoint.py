def unstandardize_multiple_linreg_parameters(zbeta0, zbeta, mu_X, mu_y, sigma, sigma_X, sigma_y):
    """Return standardized coefficients to magnitudes on raw scale.
    
    Args:
    
    Returns:
    
    """
    
    beta0 = zbeta0 * sigma_y + mu_y - np.sum(zbeta.T * mu_X / sigma_X, axis=1) * sigma_y
    beta = (zbeta / sigma_X) * sigma_y
    sigma = (sigma * sigma_y)
    
    return beta0, beta, sigma