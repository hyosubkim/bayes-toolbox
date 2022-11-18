def gamma_shape_rate_from_mode_sd(mode, sd):
    """Calculate Gamma shape and rate from mode and sd.
    
    """
    
    rate = (mode + np.sqrt(mode**2 + 4 * sd**2 )) / (2 * sd**2)
    shape = 1 + mode * rate
    return shape, rate