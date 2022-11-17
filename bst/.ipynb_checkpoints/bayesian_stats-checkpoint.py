import aesara.tensor as at
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from aesara import tensor as at


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
        return X_s, mu_X, sigma_X
    else:
        X_s = (X - X.mean()) / X.std()
        return X_s, X.mean(), X.std()



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


def BEST_copy(y, group, n_draws=1000):
    """Implementation of John Kruschke's BEST test.
    
    Compares outcomes from two groups and estimates parameters. This version
    uses smarter indexing than original. 
    
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
    
    # Calculate pooled empirical mean and SD of data to scale hyperparameters
    mu_y = y.mean()
    sigma_y = y.std()
                                                                     
    with pm.Model() as model:
        # Define priors. Arbitrarily set hyperparameters to the pooled 
        # empirical mean of data and twice pooled empirical SD, which 
        # applies very diffuse and unbiased info to these quantities. 
        group_mean = pm.Normal("group_mean", mu=mu_y, sigma=sigma_y * 2, shape=len(level))
        group_std = pm.Uniform("group_std", lower=sigma_y / 10, upper=sigma_y * 10, shape=len(level))
        
        # See Kruschke Ch 16.2.1 for in-depth rationale for prior on nu. The addition of 1 is to shift the
        # distribution so that the range of possible values of nu are 1 to infinity (with mean of 30).
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))
        
        # Define likelihood
        likelihood = pm.StudentT("likelihood", nu=nu, mu=group_mean[group_idx], sigma=group_std[group_idx], observed=y)
        
        # Contrasts of interest
        diff_of_means = pm.Deterministic("difference of means", group_mean[0] - group_mean[1])
        diff_of_stds = pm.Deterministic("difference of stds", group_std[0] - group_std[1])
        effect_size = pm.Deterministic(
            "effect size", diff_of_means / np.sqrt((group_std[0]**2 + group_std[1]**2) / 2)
        )
        
        # Sample from posterior
        idata = pm.sample(draws=n_draws)
        
    return model, idata


def robust_linear_regression(x, y, n_draws=1000):
    """Perform a robust linear regression with one predictor.
    
    Args:
        x (ndarray): The standardized predictor (independent) variable.
        y (ndarray): The standardized outcome (dependent) variable.
        n_draws: Number of random samples to draw from the posterior.
        
    Returns: 
        PyMC Model and InferenceData objects.
    """
    
    # Make sure variables are standardized (within reasonable precision). 
    assert (is_standardized(x)) & (is_standardized(y)), f"Inputs must be standardized."

    with pm.Model() as model:
        # Define priors
        beta0 = pm.Normal('beta0', mu=0, sigma=2)
        beta1 = pm.Normal('beta1', mu=0, sigma=2)
        
        sigma = pm.Uniform('sigma', 10**-3, 10**3)
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))
        
        mu = beta0 + beta1 * x

        # Define likelihood 
        likelihood = pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, observed=y)
        
        # Sample from posterior
        idata = pm.sample(draws=n_draws)
    
    return model, idata


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


def hierarchical_regression(x, y, subj):  
    """
    
    """
    
    # Convert subject variable to categorical dtype if it is not already
    if pd.api.types.is_categorical_dtype(subj):
        pass
    else:
        subj = subj.astype('category')
    subj_idx = subj.cat.codes.values
    subj_levels = subj.cat.categories
    n_subj = len(subj_levels)
    
    with pm.Model() as model:
        # Hyperpriors
        beta0 = pm.Normal('beta0', mu=0, tau=1/10**2)
        beta1 = pm.Normal('beta1', mu=0, tau=1/10**2)
        sigma0 = pm.Uniform('sigma0', 10**-3, 10**3)
        sigma1 = pm.Uniform('sigma1', 10**-3, 10**3)

        # The intuitive parameterization results in a lot of divergences.
        #beta0_s = pm.Normal('beta0_s', mu=beta0, sd=sigma0, shape=n_subj)
        #beta1_s = pm.Normal('beta1_s', mu=beta1, sd=sigma1, shape=n_subj)
        
        # See: http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/
        # for rationale of following reparameterization
        beta0_s_offset = pm.Normal('beta0_s_offset', mu=0, sd=1, shape=n_subj)
        beta0_s = pm.Deterministic('beta0_s', beta0 + beta0_s_offset * sigma0)

        beta1_s_offset = pm.Normal('beta1_s_offset', mu=0, sd=1, shape=n_subj)
        beta1_s = pm.Deterministic('beta1_s', beta1 + beta1_s_offset * sigma1)

        mu =  beta0_s[subj_idx] + beta1_s[subj_idx] * zx3

        sigma = pm.Uniform('sigma', 10**-3, 10**3)
        nu = pm.Exponential('nu', 1/29.)

        likelihood = pm.StudentT('likelihood', nu, mu=mu, sd=sigma, observed=zy3)  
        
        # Sample from posterior
        idata = pm.sample(draws=n_draws)
        
        return model, idata

    
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

        

def multiple_linear_regression(X, y, n_draws=1000):
    """Perform a Bayesian multiple linear regression.
    
    Args:
        X (dataframe): Predictor variables are in different columns.
        y (ndarray/Series): The outcome variable.
        
    Returns:
    
    """
    
    # Standardize both predictor and outcome variables.
    if (is_standardized(X)) & (is_standardized(y)):
        pass
    else:
        X, _, _ = standardize(X)
        y, _, _ = standardize(y)
    
    # For explanation of how predictor variables are handled, see:
    # https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html
    with pm.Model(coords={"predictors": X.columns.values}) as model:
        # Define priors
        beta0 = pm.Normal("beta0", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2, dims="predictors")

        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Exponential("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))
        
        mu = beta0 + pm.math.dot(X, beta)
        sigma = pm.Uniform("sigma", 10**-5, 10)
        
        # Define likelihood
        likelihood = pm.StudentT("likelihood", nu=nu, mu=mu, lam=1/sigma**2, observed=y)
    
        # Sample the posterior
        idata = pm.sample(draws=n_draws)
        
        return model, idata
    
    
def gamma_shape_rate_from_mode_sd(mode, sd):
    """Calculate Gamma shape and rate from mode and sd.

    """

    rate = (mode + np.sqrt(mode**2 + 4 * sd**2 )) / (2 * sd**2)
    shape = 1 + mode * rate
    return shape, rate


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
    
    # First, check to see if passed variable is of type "category".
    if pd.api.types.is_categorical_dtype(x):
        pass
    else:
        x = x.astype('category')
    categorical_values = x.cat.codes.values
    levels = x.cat.categories
    n_levels = len(levels)

    return categorical_values, levels, n_levels


def metric_outcome_one_nominal_predictor(x, y, mu_y, sigma_y, n_draws=1000):
    """
    
    """
    x_vals, levels, n_levels = parse_categorical(x)
    
    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)
    
    with pm.Model(coords={"groups": levels}) as model:
        # 'a' indicates coefficients not yet obeying sum-to-zero contraint
        sigma_a = pm.Gamma('sigma_a', alpha=a_shape, beta=a_rate)
        a0 = pm.Normal('a0', mu=mu_y, sigma=sigma_y * 5)
        a = pm.Normal('a', 0.0, sigma=sigma_a, dims="groups")

        sigma_y = pm.Uniform('sigma_y', sigma_y / 100, sigma_y * 10)
        likelihood = pm.Normal('likelihood', a0 + a[x_vals], sigma=sigma_y, observed=y)

        # Convert a0, a to sum-to-zero b0,b 
        m = pm.Deterministic('m', a0 + a)
        b0 = pm.Deterministic('b0', at.mean(m))
        b = pm.Deterministic('b', m - b0) 
        
        idata = pm.sample(draws=n_draws)

        return model, idata
    
    
def metric_outcome_one_nominal_one_metric_predictor(x, x_met, y, mu_x_met, mu_y, sigma_x_met, sigma_y, n_draws=1000):
    """

    """
    x_vals, levels, n_levels = parse_categorical(x)

    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)

    with pm.Model(coords={"groups": levels}) as model:
        # 'a' indicates coefficients not yet obeying sum-to-zero contraint
        sigma_a = pm.Gamma('sigma_a', alpha=a_shape, beta=a_rate)
        a0 = pm.Normal('a0', mu=mu_y, sigma=sigma_y * 5)
        a = pm.Normal('a', 0.0, sigma=sigma_a, dims="groups")
        a_met = pm.Normal("a_met", mu=0, sigma=2 * sigma_y / sigma_x_met)
        # Note that in Warmhoven notebook he uses SD of residuals to set
        # lower bound on sigma_y
        sigma_y = pm.Uniform('sigma_y', sigma_y / 100, sigma_y * 10)
        mu = a0 + a[x_vals] + a_met * (x_met - mu_x_met)
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma_y, observed=y)

        # Convert a0, a to sum-to-zero b0, b 
        b0 = pm.Deterministic('b0', a0 + at.mean(a) + a_met * (-mu_x_met))
        b = pm.Deterministic('b', a - at.mean(a)) 

        idata = pm.sample(draws=n_draws)

        return model, idata
    
    
def robust_anova(x, y, mu_y, sigma_y, n_draws=1000):
    """
    
    """
    x_vals, levels, n_levels = parse_categorical(x)
    
    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)
    
    with pm.Model(coords={"groups": levels}) as model:
        # 'a' indicates coefficients not yet obeying sum-to-zero contraint
        sigma_a = pm.Gamma('sigma_a', alpha=a_shape, beta=a_rate)
        a0 = pm.Normal('a0', mu=mu_y, sigma=sigma_y * 10)
        a = pm.Normal('a', 0.0, sigma=sigma_a, dims="groups")
        # Hyperparameters 
        sigma_y_sd = pm.Gamma('sigma_y_sd', alpha=a_shape, beta=a_rate)
        sigma_y_mode = pm.Gamma("sigma_y_mode", alpha=a_shape, beta=a_rate)
        sigma_y_rate = (sigma_y_mode + np.sqrt(sigma_y_mode**2 + 4 * sigma_y_sd**2)) / 2 * sigma_y_sd**2
        sigma_y_shape = sigma_y_mode * sigma_y_rate
        
        sigma_y = pm.Gamma("sigma", alpha=sigma_y_shape, beta=sigma_y_rate, dims="groups")
        nu_minus1 = pm.Exponential('nu_minus1', 1 / 29)
        nu = pm.Deterministic('nu', nu_minus1 + 1)
        likelihood = pm.Normal('likelihood', a0 + a[x_vals], sigma=sigma_y[x_vals], observed=y)

        # Convert a0, a to sum-to-zero b0, b 
        m = pm.Deterministic('m', a0 + a)
        b0 = pm.Deterministic('b0', at.mean(m))
        b = pm.Deterministic('b', m - b0) 
        
        # Initialization argument is necessary for sampling to converge
        idata = pm.sample(draws=n_draws, init='advi+adapt_diag')

        return model, idata
    

def two_factor_anova(x1, x2, y):
    
    mu_y = y.mean()
    sigma_y = y.std()

    x1 = df.Pos
    x2 = df.Org

    a_shape, a_rate = bst.gamma_shape_rate_from_mode_sd(sigma_y / 2 , 2 * sigma_y)
    x1_vals, levels1, n_levels1 = bst.parse_categorical(x1)
    x2_vals, levels2, n_levels2 = bst.parse_categorical(x2)

    with pm.Model(coords={"rank": levels1, "dept": levels2}) as model:

        #a0 = pm.Normal('a0', mu_y, tau=1/(sigma_y*5)**2)
        a0_tilde = pm.Normal('a0_tilde', mu=0, sigma=1)
        a0 = pm.Deterministic('a0', mu_y + sigma_y * 5 * a0_tilde)

        sigma_a1 = pm.Gamma('sigma_a1', a_shape, a_rate)
        #a1 = pm.Normal('a1', 0.0, tau=1/sigma_a1**2, shape=n_levels1)
        a1_tilde = pm.Normal('a1_tilde', mu=0, sigma=1, dims="rank")
        a1 = pm.Deterministic('a1', 0.0 + sigma_a1*a1_tilde)

        sigma_a2 = pm.Gamma('sigma_a2', a_shape, a_rate)
        #a2 = pm.Normal('a2', 0.0, tau=1/sigma_a2**2, shape=n_levels2)
        a2_tilde = pm.Normal('a2_tilde', mu=0, sigma=1, dims="dept")
        a2 = pm.Deterministic('a2', 0.0 + sigma_a2*a2_tilde)

        sigma_a1a2 = pm.Gamma('sigma_a1a2', a_shape, a_rate)
        #a1a2 = pm.Normal('a1a2', 0.0, 1/sigma_a1a2**2, shape=(n_levels1, n_levels2))
        a1a2_tilde = pm.Normal('a1a2_tilde', mu=0, sigma=1, dims=("rank", "dept"))
        a1a2 = pm.Deterministic('a1a2', 0.0 + sigma_a1a2*a1a2_tilde)

        mu = a0 + a1[x1_vals] + a2[x2_vals] +a1a2[x1_vals, x2_vals]
        sigma = pm.Uniform('sigma', sigma_y / 100, sigma_y * 10)

        likelihood = pm.Normal('likelihood', mu, sigma=sigma, observed=y) 

        idata = pm.sample(nuts={'target_accept': 0.95})


def mixed_model_anova(between_subj_var, within_subj_var, subj_id, y, n_samples=1000):

    # Statistical model: Split-plot design after Kruschke Ch. 20
    # Between-subjects factor (i.e., group)
    x_between, levels_x_between, num_levels_x_between = parse_categorical(between_subj_var)
  
    # Within-subjects factor (i.e., target set)
    x_within, levels_x_within, num_levels_x_within = parse_categorical(within_subj_var)

    # Individual subjects
    x_subj, levels_x_subj, num_levels_x_subj = parse_categorical(subj_id)

    # Dependent varia_ble 
    mu_y = y.mean()
    sigma_y = y.std()

    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2 , 2 * sigma_y)

    with pm.Model(coords={
        "between_subj": levels_x_between, 
        "within_subj": levels_x_within,
        "subj": levels_x_subj}
        ) as model:

        # Baseline value
        a0 = pm.Normal('a0', mu=mu_y, sigma=sigma_y * 5)

        # Deflection from baseline for between subjects factor
        sigma_B = pm.Gamma('sigma_B', a_shape, a_rate)
        aB = pm.Normal('aB', mu=0.0, sigma=sigma_B, dims="between_subj")

        # Deflection from baseline for within subjects factor
        sigma_W = pm.Gamma('sigma_W', a_shape, a_rate)
        aW = pm.Normal('aW', mu=0.0, sigma=sigma_W, dims="within_subj")

        # Deflection from baseline for combination of between and within subjects factors
        sigma_BxW = pm.Gamma('sigma_BxW', a_shape, a_rate)
        aBxW = pm.Normal('aBxW', mu=0.0, sigma=sigma_BxW, dims=("between_subj", "within_subj"))

        # Deflection from baseline for individual subjects
        sigma_S = pm.Gamma('sigma_S', a_shape, a_rate)
        aS = pm.Normal('aS', mu=0.0, sigma=sigma_S, dims="subj")

        mu = a0 + aB[x_between] + aW[x_within] + aBxW[x_between, x_within] + aS[x_subj] 
        sigma = pm.Uniform('sigma', lower=sigma_y / 100, upper=sigma_y * 10)

        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(draws=n_samples, target_accept=.95, tune=2000)
    
    return model, idata