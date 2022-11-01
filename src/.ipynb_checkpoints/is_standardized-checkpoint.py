def is_standardized(x, eps=0.0001):
    """Checks to see if variable is standardized (i.e., N(0, 1)).
    
    Args: 
        x: Random variable.
        eps: Some small value to represent tolerance level.
    
    Returns:
        Bool
    """
    
    return (x.mean()**2 < eps) & ((x.std() - 1)**2 < eps)
