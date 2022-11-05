def unstandardize_linreg_parameters(zeta0, zeta1, mu_x, mu_y, sigma, sigma_x, sigma_y):
    """Convert parameters back to raw scale of data.
    
    Function takes in parameter values from PyMC InferenceData and returns
    them in original scale of raw data. 
    
    Args:
        zeta0 (): Intercept for standardized data.
        zeta1 (PyMC trace): Slope for standardized data.
        mu_y (scalar): Mean of outcome variable.
        sigma_x (scalar): SD of predictor variable.
        sigma_y (scalar): SD of outcome variable.
        
    Returns:
        ndarrays
    """
    
    beta0 = zeta0*sigma_y + mu_y - zeta1*mu_x*sigma_y/sigma_x
    beta1 = zeta1*sigma_y/sigma_x
    sigma = (sigma * sigma_y)
   
    return beta0, beta1, sigma