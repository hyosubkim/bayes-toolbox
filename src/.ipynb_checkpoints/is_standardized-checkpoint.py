def is_standardized(X, eps=0.0001):
    """Checks to see if variable is standardized (i.e., N(0, 1)).
    
    Args: 
        x: Random variable.
        eps: Some small value to represent tolerance level.
    
    Returns:
        Bool
    """
    
    if isinstance(X, pd.DataFrame):
        mu_X = X.mean().values
        sigma_X = X.std().values
        X_s = ((X - mu_X) / sigma_X)
        return np.equal((mu_X**2 < eps).sum() + ((sigma_X - 1)**2 < eps).sum(),
                        len(mu_X) + len(sigma_X))
    else:
        return (X.mean()**2 < eps) & ((X.std() - 1)**2 < eps)
