def standardize(X):
    """Standardizes the input variable.
    
    Args:
        X (ndarray/Series/DataFrame): The variable(s) to standardize. 
    
    Returns:
        ndarray/Series/DataFrame: The standardized version of the input X.
    """
    
    # Check if X is a dataframe, in which case it will be handled differently. 
    if isinstance(X, pd.DataFrame):
        mu_X = X.mean().values
        sigma_X = X.std().values
        X_s = ((X - mu_X) / sigma_X)
        return X_s
    else:
        return (X - X.mean()) / X.std()
