import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns


def standardize(x):
    
    return (x - x.mean()) / x.std()


def compare_one_group(y, group=None, n_samples=1000):
    
    # Check to see if group variable was passed. If so, then this means the goal is to compare
    # difference scores on a within subjects variable (e.g., time). Otherwise, we are comparing 
    # location parameter to zero.
    if group is None:
        pass
    else:
        group = group.astype('category')
        level = group.cat.categories
        assert len(level) == 2, f"Expected two groups but got {len(level)}"
        y = y[group==level[1]].to_numpy() - y[group==level[0]].to_numpy()
    
    with pm.Model() as model:
        # Set priors
        mu = pm.Normal('mu', mu=y.mean(), sd=y.std()*100)
        sigma = pm.Uniform('sigma', y.std()/1000, y.std()*1000)
        nu_minus1 = pm.Exponential('nu_minus_one', 1/29)
        nu = pm.Deterministic('nu', nu_minus1 + 1)

        like = pm.StudentT('like', nu, mu, sd=sigma, observed=y)

        # Sample from posterior
        idata = pm.sample(return_inferencedata=True)
        
    return idata


def compare_two_groups(y, group, sigma_low, sigma_high, n_draws=1000):
    
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
    
    # Arbitrarily set hyperparameters to the pooled empirical mean of data and twice pooled empirical SD, 
    # which applies very diffuse info to these quantities and does not favor one or the other a priori
    mu_m = y.mean()
    mu_s = y.std() * 2

    with pm.Model() as model:
        # Set priors
        group1_mean = pm.Normal("group1_mean", mu=mu_m, sigma=mu_s)
        group2_mean = pm.Normal("group2_mean", mu=mu_m, sigma=mu_s)
        group1_std = pm.Uniform("group1_std", lower=sigma_low, upper=sigma_high)
        group2_std = pm.Uniform("group2_std", lower=sigma_low, upper=sigma_high)
        
        # See Kruschke Ch 16.2.1 for in-depth rationale for prior on nu. Briefly, for values of nu greater than
        # or equal to 30, the distribution converges to a standard normal. Since we want to allow for smaller and
        # larger values of nu, we are setting the prior to be an exponential with a mean of 30 (i.e., there is
        # substantial probability for smaller values of nu). The single parameter for an exponential is the reciprocal
        # of the mean. The addition of 1 is to shift the distribution so that possible possible values of nu are 1 to 
        # infinity (and the mean of the distribution is 30).
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        
        # Need to convert SD to precision for model
        lambda_1 = group1_std**-2
        lambda_2 = group2_std**-2
        
        # Likelihood
        like1 = pm.StudentT("group1", nu=nu, mu=group1_mean, lam=lambda_1, observed=y_group1)
        like2 = pm.StudentT("group2", nu=nu, mu=group2_mean, lam=lambda_2, observed=y_group2)
        
        # Contrasts
        diff_of_means = pm.Deterministic("difference of means", group1_mean - group2_mean)
        diff_of_stds = pm.Deterministic("difference of stds", group1_std - group2_std)
        effect_size = pm.Deterministic(
            "effect size", diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2)
        )
        
        # Sample from posterior
        idata = pm.sample(draws=n_draws, return_inferencedata=True)
        
    return model, idata


def robust_linear_regression():
    
    with pm.Model() as model:
        beta0 = pm.Normal('beta0', mu=0, tau=1/10**2)
        beta1 = pm.Normal('beta1', mu=0, tau=1/10**2)
        mu = beta0 + beta1*x

        sigma = pm.Uniform('sigma', 10**-3, 10**3)
        nu = pm.Exponential('nu', 1/29.0)

        likelihood = pm.StudentT('likelihood', nu, mu=mu, sd=sigma, observed=y)

        idata = pm.sample(return_inferencedata=True)
    
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

        

    