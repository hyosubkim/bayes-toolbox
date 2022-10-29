import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns


def standardize(x):
    """Standardizes the input variable.
    
    Args:
        x (ndarray/Series): the variable to standardize 
    
    Returns:
        ndarray/Series: the standardized version of the input variable x
    """
        
    return (x - x.mean()) / x.std()


def BEST_paired(y1, y2=None, n_samples=1000):
    """BEST procedure on single sample or paired samples. 
    
    Args: 
        y1 (ndarray/Series): Either single sample or difference scores. 
        y2 (ndarray/Series): (Optional) If provided, represents the paired 
          sample (i.e., y2 elements are in same order as y1).
    Returns: 
        PyMC Model and InferenceData objects.
    """
    
    # Check to see if block variable was passed. If so, then this means the
    # goal is to compare difference scores on a within subjects variable 
    # (e.g., block). Otherwise, we are comparing location parameter to zero.
    if y2 is None:
        pass
    else:
        assert len(y1) == len(y2), f"There must be equal numbers of observations."
        # Convert pre and post to difference scores.
        y = y1 - y2
    
    # Calculate pooled empirical mean and SD of data to scale hyperparameters
    mu_y = y.mean()
    sigma_y = y.std()
                                                                     
    with pm.Model() as model:
        # Define priors
        mu = pm.Normal('mu', mu=mu_y, sd=sigma_y * 10)
        sigma = pm.Uniform('sigma', sigma_y / 10, sigma_y * 10)
        nu_minus1 = pm.Exponential('nu_minus_one', 1 / 29)
        nu = pm.Deterministic('nu', nu_minus1 + 1)
        
        # Define likelihood
        likelihood = pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, observed=y)

        # Sample from posterior
        idata = pm.sample(draws=n_draws)
        
    return model, idata


def BEST(y, group, n_draws=1000):
    """Implementation of John Kruschke's BEST test.
    
    Compares outcomes from two groups and estimates parameters.
    
    Args:
        y (ndarray/Series): The metric outcome variable.
        group: The grouping variable providing that indexes into y.
        n_draws: Number of random samples to draw from the posterior.
    
    Returns: 
        PyMC Model and InferenceData objects.
    """
    
    # Convert grouping variable to categorical dtype if it is not already
    if pd.api.types.is_categorical_dtype(group):
        pass
    else:
        group = group.astype('category')
        
    # Extract group levels and make sure there are only two
    level = group.cat.categories
    assert len(level) == 2, f"Expected two groups but got {len(level)}."
    
    # Split observations by group
    y_group1 = y[group==level[0]]
    y_group2 = y[group==level[1]]
    
    # Calculate pooled empirical mean and SD of data to scale hyperparameters
    mu_y = y.mean()
    sigma_y = y.std()
    
    # Arbitrarily set hyperparameters to the pooled empirical mean of data and 
    # twice pooled empirical SD, which applies very diffuse info to these 
    # quantities and does not favor one or the other a priori
    mu_m = mu_y
    mu_s = sigma_y * 2
                                                                     
    with pm.Model() as model:
        # Define priors
        group1_mean = pm.Normal("group1_mean", mu=mu_m, sigma=mu_s)
        group2_mean = pm.Normal("group2_mean", mu=mu_m, sigma=mu_s)
        group1_std = pm.Uniform("group1_std", lower=sigma_y / 10, upper=sigma_y * 10)
        group2_std = pm.Uniform("group2_std", lower=sigma_y / 10, upper=sigma_y * 10)
        
        # See Kruschke Ch 16.2.1 for in-depth rationale for prior on nu. The addition of 1 is to shift the
        # distribution so that the range of possible values of nu are 1 to infinity (with mean of 30).
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))
        
        # Define likelihood
        likelihood1 = pm.StudentT("group1", nu=nu, mu=group1_mean, sigma=group1_std, observed=y_group1)
        likelihood2 = pm.StudentT("group2", nu=nu, mu=group2_mean, sigma=group2_std, observed=y_group2)
        
        # Contrasts of interest
        diff_of_means = pm.Deterministic("difference of means", group1_mean - group2_mean)
        diff_of_stds = pm.Deterministic("difference of stds", group1_std - group2_std)
        effect_size = pm.Deterministic(
            "effect size", diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2)
        )
        
        # Sample from posterior
        idata = pm.sample(draws=n_draws)
        
    return model, idata


def robust_linear_regression(n_draws=1000):
    
    with pm.Model() as model:
        # Define priors
        beta0 = pm.Normal('beta0', mu=0, tau=1/10**2)
        beta1 = pm.Normal('beta1', mu=0, tau=1/10**2)
        mu = beta0 + beta1*x

        sigma = pm.Uniform('sigma', 10**-3, 10**3)
        nu = pm.Exponential('nu', 1/29.0)

        likelihood = pm.StudentT('likelihood', nu, mu=mu, sd=sigma, observed=y)

        idata = pm.sample(draws=n_draws)
    
    return idata


def hierarchical_regression():  
    
    with pm.Model() as model:
        # Hyperpriors
        beta0 = pm.Normal('beta0', mu=0, tau=1/10**2)
        beta1 = pm.Normal('beta1', mu=0, tau=1/10**2)
        sigma0 = pm.Uniform('sigma0', 10**-3, 10**3)
        sigma1 = pm.Uniform('sigma1', 10**-3, 10**3)

        # The below parameterization resulted in a lot of divergences.
        #beta0_s = pm.Normal('beta0_s', mu=beta0, sd=sigma0, shape=n_subj)
        #beta1_s = pm.Normal('beta1_s', mu=beta1, sd=sigma1, shape=n_subj)
        
        beta0_s_offset = pm.Normal('beta0_s_offset', mu=0, sd=1, shape=n_subj)
        beta0_s = pm.Deterministic('beta0_s', beta0 + beta0_s_offset * sigma0)

        beta1_s_offset = pm.Normal('beta1_s_offset', mu=0, sd=1, shape=n_subj)
        beta1_s = pm.Deterministic('beta1_s', beta1 + beta1_s_offset * sigma1)

        mu =  beta0_s[subj_idx] + beta1_s[subj_idx] * zx3

        sigma = pm.Uniform('sigma', 10**-3, 10**3)
        nu = pm.Exponential('nu', 1/29.)

        likelihood = pm.StudentT('likelihood', nu, mu=mu, sd=sigma, observed=zy3)  
        
        # Sample from posterior
        idata = pm.sample(return_inferencedata=True)

        
def BEST_copy(y, group, n_draws=1000):
    """Implementation of John Kruschke's BEST test.
    
    Compares outcomes from two groups and estimates parameters.
    
    Args:
        y (ndarray/Series): The metric outcome variable.
        group: The grouping variable providing that indexes into y.
        n_draws: Number of random samples to draw from the posterior.
    
    Returns: 
        PyMC Model and InferenceData objects.
    """
    
    # Convert grouping variable to categorical dtype if it is not already
    if pd.api.types.is_categorical_dtype(group):
        pass
    else:
        group = group.astype('category')
    group_idx = group.cat.codes.values
        
    # Extract group levels and make sure there are only two
    level = group.cat.categories
    assert len(level) == 2, f"Expected two groups but got {len(level)}."
    
    # Split observations by group
    y_group1 = y[group==level[0]]
    y_group2 = y[group==level[1]]
    
    # Calculate pooled empirical mean and SD of data to scale hyperparameters
    mu_y = y.mean()
    sigma_y = y.std()
    
    # Arbitrarily set hyperparameters to the pooled empirical mean of data and 
    # twice pooled empirical SD, which applies very diffuse info to these 
    # quantities and does not favor one or the other a priori
    mu_m = mu_y
    mu_s = sigma_y * 2
                                                                     
    with pm.Model() as model:
        # Define priors
        group_mean = pm.Normal("group_mean", mu=mu_m, sigma=mu_s, shape=len(level))
        # group2_mean = pm.Normal("group2_mean", mu=mu_m, sigma=mu_s)
        group_std = pm.Uniform("group_std", lower=sigma_y / 10, upper=sigma_y * 10, shape=len(level))
        # group2_std = pm.Uniform("group2_std", lower=sigma_y / 10, upper=sigma_y * 10)
        
        # See Kruschke Ch 16.2.1 for in-depth rationale for prior on nu. The addition of 1 is to shift the
        # distribution so that the range of possible values of nu are 1 to infinity (with mean of 30).
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))
        
        # Define likelihood
        likelihood = pm.StudentT("group1", nu=nu, mu=group_mean[group_idx], sigma=group_std[group_idx], observed=y)
        # likelihood2 = pm.StudentT("group2", nu=nu, mu=group2_mean, sigma=group2_std, observed=y_group2)
        
        # Contrasts of interest
        diff_of_means = pm.Deterministic("difference of means", group_mean[0] - group_mean[1])
        diff_of_stds = pm.Deterministic("difference of stds", group_std[0] - group_std[1])
        effect_size = pm.Deterministic(
            "effect size", diff_of_means / np.sqrt((group_std[0]**2 + group_std[1]**2) / 2)
        )
        
        # Sample from posterior
        idata = pm.sample(draws=n_draws)
        
    return model, idata

    