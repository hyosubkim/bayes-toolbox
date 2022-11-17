def unstandardize_multiple_linreg_parameters(zbeta0, zbeta, mu_X, mu_y, sigma, sigma_X, sigma_y):
    """Rescale standardized coefficients to magnitudes on raw scale.
    
    If the posterior samples come from multiple chains, they should be combined
    and the zbetas should have dimensionality of (predictors, draws). 
    
    Args:
        zbeta0: Standardized intercept.
        zbeta: Standardized multiple regression coefficients for predictor
                variables.
        mu_X: Mean of predictor variables.
        mu_y: Mean of outcome variable.
        sigma: SD of likelihood on standardized scale.
        sigma_X: SD of predictor variables.
        sigma_y: SD of outcome variable.
        
    Returns:
    
    """
    # beta0 will turn out to be 1d bc we are broadcasting, and Numpy's sum 
    # reduces the axis over which summation occurs.
    beta0 = zbeta0 * sigma_y + mu_y - np.sum(zbeta.T * mu_X / sigma_X, axis=1) * sigma_y
    beta = (zbeta.T / sigma_X) * sigma_y
    sigma = (sigma * sigma_y)
    
    return beta0, beta, sigma