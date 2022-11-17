def parse_categorical(x):
    """A function for extracting information from a grouping variable.
    
    If the input arg is not already a category-type variable, convert 
    it to one. Then, extract the codes, unique levels, and number of 
    levels from the variable.
    
    Args:
        x (categorical): The categorical type variable to parse.
    
    Returns:
        ndarrays for the values, unique level, and number of levels.
    """
    
    # First check to see
    if pd.api.types.is_categorical_dtype(x):
        pass
    else:
        x = x.astype('category')
    categorical_values = x.cat.codes.values
    levels = x.cat.categories
    n_levels = len(levels)

    return categorical_values, levels, n_levels