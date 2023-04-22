# MIT License

# Copyright (c) 2022 Hyosub Kim

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Contains code copied and adapted from Jordi Warmenhoven's GitHub repository:
# https://github.com/JWarmenhoven/DBDA-python (see LICENSE).

"""
A collection of Bayesian statistical models and associated utility functions.
"""

import aesara.tensor as at
import arviz as az
import numpy as np
import numpy.ma as ma
import pandas as pd
import pymc as pm


def standardize(X):
    """Standardize the input variable.

    Args:
        X (ndarray): The variable(s) to standardize.

    Returns:
        ndarray
    """

    # Check if X is a dataframe, in which case it will be handled differently.
    if isinstance(X, pd.DataFrame):
        mu_X = X.mean().values
        sigma_X = X.std().values
    else:
        mu_X = X.mean(axis=0)
        sigma_X = X.std(axis=0)
    
    # Standardize X
    X_s = (X - mu_X) / sigma_X
    
    return X_s, mu_X, sigma_X


def parse_categorical(x):
    """A function for extracting information from a grouping variable.

    If the input arg is not already a category-type variable, converts
    it to one. Then, extracts the codes, unique levels, and number of
    levels from the variable.

    Args:
        x (categorical): The categorical type variable to parse.

    Returns:
        The codes, unique levels, and number of levels from the input variable.
    """

    # First, check to see if passed variable is of type "category".
    if pd.api.types.is_categorical_dtype(x):
        pass
    else:
        x = x.astype("category")
    categorical_values = x.cat.codes.values
    levels = x.cat.categories
    n_levels = len(levels)

    return categorical_values, levels, n_levels


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
        X_s = (X - mu_X) / sigma_X
        return np.equal(
            (mu_X**2 < eps).sum() + ((sigma_X - 1) ** 2 < eps).sum(),
            len(mu_X) + len(sigma_X),
        )
    else:
        return (X.mean() ** 2 < eps) & ((X.std() - 1) ** 2 < eps)


def BEST(y, group, n_draws=1000):
    """Implementation of John Kruschke's BEST test.

    Estimates parameters related to outcomes of two groups. See:
        https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf
        for more details.

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
        group = group.astype("category")
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
        group_mean = pm.Normal(
            "group_mean", mu=mu_y, sigma=sigma_y * 2, shape=len(level)
        )
        group_std = pm.Uniform(
            "group_std", lower=sigma_y / 10, upper=sigma_y * 10, shape=len(level)
        )

        # See Kruschke Ch 16.2.1 for in-depth rationale for prior on nu. The addition 
        # of 1 is to shift the distribution so that the range of possible values of nu
        # are 1 to infinity (with mean of 30).
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

        # Define likelihood
        likelihood = pm.StudentT(
            "likelihood",
            nu=nu,
            mu=group_mean[group_idx],
            sigma=group_std[group_idx],
            observed=y,
        )

        # Contrasts of interest
        diff_of_means = pm.Deterministic(
            "difference of means", group_mean[0] - group_mean[1]
        )
        diff_of_stds = pm.Deterministic(
            "difference of stds", group_std[0] - group_std[1]
        )
        effect_size = pm.Deterministic(
            "effect size",
            diff_of_means / np.sqrt((group_std[0] ** 2 + group_std[1] ** 2) / 2),
        )

        # Sample from posterior
        idata = pm.sample(draws=n_draws)

    return model, idata


def BEST_paired(y1, y2=None, n_draws=1000):
    """BEST procedure on single or paired sample(s).

    Args:
        y1 (ndarray/Series): Either single sample or difference scores.
        y2 (ndarray/Series): (Optional) If provided, represents the paired
          sample (i.e., y2 elements are in same order as y1).
    Returns:
        PyMC Model and InferenceData objects.
    """

    # Check to see if y2 was entered. If so, then this means the
    # goal is to compare difference scores on a within subjects variable
    # (e.g., block). Otherwise, we are comparing location parameter to zero.
    if y2 is None:
        y = y1
    else:
        assert len(y1) == len(y2), f"There must be equal numbers of observations."
        # Convert pre and post to difference scores.
        y = y1 - y2

    # Calculate pooled empirical mean and SD of data to scale hyperparameters
    mu_y = y.mean()
    sigma_y = y.std()

    with pm.Model() as model:
        # Define priors
        mu = pm.Normal("mu", mu=mu_y, sigma=sigma_y * 10)
        sigma = pm.Uniform("sigma", sigma_y / 10, sigma_y * 10)
        nu_minus1 = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Deterministic("nu", nu_minus1 + 1)

        # Define likelihood
        likelihood = pm.StudentT("likelihood", nu=nu, mu=mu, sigma=sigma, observed=y)

        # Standardized effect size
        effect_size = pm.Deterministic("effect_size", mu / sigma)

        # Sample from posterior
        idata = pm.sample(draws=n_draws)

    return model, idata


def robust_linear_regression(x, y, n_draws=1000):
    """Perform a robust linear regression with one predictor.

    The observations are modeled with a t-distribution which can much more
    readily account for "outlier" observations.

    Args:
        x (ndarray): The standardized predictor (independent) variable.
        y (ndarray): The standardized outcome (dependent) variable.
        n_draws: Number of random samples to draw from the posterior.

    Returns:
        PyMC Model and InferenceData objects.
    """

    # Standardize both predictor and outcome variables.
    if (is_standardized(x)) & (is_standardized(y)):
        pass
    else:
        x, _, _ = standardize(x)
        y, _, _ = standardize(y)

    with pm.Model() as model:
        # Define priors
        zbeta0 = pm.Normal("zbeta0", mu=0, sigma=2)
        zbeta1 = pm.Normal("zbeta1", mu=0, sigma=2)

        sigma = pm.Uniform("sigma", 10**-3, 10**3)
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

        mu = zbeta0 + zbeta1 * x

        # Define likelihood
        likelihood = pm.StudentT("likelihood", nu=nu, mu=mu, sigma=sigma, observed=y)

        # Sample from posterior
        idata = pm.sample(draws=n_draws)

    return model, idata


def unstandardize_linreg_parameters(zbeta0, zbeta1, sigma, x, y):
    """Convert parameters back to raw scale of data.

    Function takes in parameter values from PyMC InferenceData and returns
    them in original scale of raw data.

    Args:
        zbeta0 (): Intercept for standardized data.
        zbeta1 (): Slope for standardized data.
        mu_y (scalar): Mean of outcome variable.
        sigma_x (scalar): SD of predictor variable.
        sigma_y (scalar): SD of outcome variable.

    Returns:
        ndarrays
    """
    _, mu_x, sigma_x = standardize(x)
    _, mu_y, sigma_y = standardize(y)

    beta0 = zbeta0 * sigma_y + mu_y - zbeta1 * mu_x * sigma_y / sigma_x
    beta1 = zbeta1 * sigma_y / sigma_x
    sigma = sigma * sigma_y

    return beta0, beta1, sigma


def hierarchical_regression(x, y, subj, n_draws=1000, acceptance_rate=0.9):
    """A multi-level model for estimating group and individual level parameters.

    Args:
        x: Predictor variable.
        y: Outcome variable.
        subj: Subj id variable.

    Returns:
        PyMC Model and InferenceData objects.
    """
    zx, mu_x, sigma_x = standardize(x)
    zy, mu_y, sigma_y = standardize(y)

    # Convert subject variable to categorical dtype if it is not already
    subj_idx, subj_levels, n_subj = parse_categorical(subj)

    # Taking advantage of the label-based indexing provided by xarray. See:
    # https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html
    with pm.Model(coords={"subj": subj_levels}) as model:
        # Hyperpriors
        zbeta0 = pm.Normal("zbeta0", mu=0, tau=1 / 10**2)
        zbeta1 = pm.Normal("zbeta1", mu=0, tau=1 / 10**2)
        zsigma0 = pm.Uniform("zsigma0", 10**-3, 10**3)
        zsigma1 = pm.Uniform("zsigma1", 10**-3, 10**3)

        # Priors for individual subject parameters. See:
        # http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/
        # for rationale of following reparameterization.
        zbeta0_s_offset = pm.Normal("zbeta0_s_offset", mu=0, sigma=1, dims="subj")
        zbeta0_s = pm.Deterministic(
            "zbeta0_s", zbeta0 + zbeta0_s_offset * zsigma0, dims="subj"
        )

        zbeta1_s_offset = pm.Normal("zbeta1_s_offset", mu=0, sigma=1, dims="subj")
        zbeta1_s = pm.Deterministic(
            "zbeta1_s", zbeta1 + zbeta1_s_offset * zsigma1, dims="subj"
        )

        mu = zbeta0_s[subj_idx] + zbeta1_s[subj_idx] * zx

        zsigma = pm.Uniform("zsigma", 10**-3, 10**3)
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Deterministic("nu", nu_minus_one + 1)

        # Define likelihood function
        likelihood = pm.StudentT("likelihood", nu, mu=mu, sigma=zsigma, observed=zy)

        # Sample from posterior
        idata = pm.sample(draws=n_draws, target_accept=acceptance_rate)

        return model, idata


def multiple_linear_regression(X, y, n_draws=1000):
    """Perform a Bayesian multiple linear regression.

    Args:
        X (dataframe): Predictor variables are in different columns.
        y (ndarray/Series): The outcome variable.

    Returns:
        PyMC Model and InferenceData objects.
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
        zbeta0 = pm.Normal("zbeta0", mu=0, sigma=2)
        zbeta = pm.Normal("zbeta", mu=0, sigma=2, dims="predictors")

        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29)
        nu = pm.Exponential("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

        mu = zbeta0 + pm.math.dot(X, zbeta)
        zsigma = pm.Uniform("zsigma", 10**-5, 10)

        # Define likelihood
        likelihood = pm.StudentT(
            "likelihood", nu=nu, mu=mu, lam=1 / zsigma**2, observed=y
        )

        # Sample the posterior
        idata = pm.sample(draws=n_draws)

        return model, idata


def unstandardize_multiple_linreg_parameters(zbeta0, zbeta, zsigma, X, y):
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
        Standardized coefficients and scale parameter.
    """

    _, mu_X, sigma_X = standardize(X)
    _, mu_y, sigma_y = standardize(y)

    # beta0 will turn out to be 1d bc we are broadcasting, and Numpy's sum
    # reduces the axis over which summation occurs.
    beta0 = zbeta0 * sigma_y + mu_y - np.sum(zbeta.T * mu_X / sigma_X, axis=1) * sigma_y
    beta = (zbeta.T / sigma_X) * sigma_y
    sigma = zsigma * sigma_y

    return beta0, beta, sigma


def gamma_shape_rate_from_mode_sd(mode, sd):
    """Calculate Gamma shape and rate parameters from mode and sd."""
    rate = (mode + np.sqrt(mode**2 + 4 * sd**2)) / (2 * sd**2)
    shape = 1 + mode * rate

    return shape, rate


def hierarchical_bayesian_anova(x, y, n_draws=1000, acceptance_rate=0.9):
    """Models metric outcome resulting from single categorical predictor.

    Args:
        x: The categorical (nominal) predictor variable.
        y: The outcome variable.

    Returns:
        PyMC Model and InferenceData objects.
    """

    mu_y = y.mean()
    sigma_y = y.std()

    x_vals, x_levels, n_levels = parse_categorical(x)
    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)

    with pm.Model(coords={"groups": x_levels}) as model:
        # 'a' indicates coefficients not yet obeying sum-to-zero contraint
        sigma_a = pm.Gamma("sigma_a", alpha=a_shape, beta=a_rate)
        a0 = pm.Normal("a0", mu=mu_y, sigma=sigma_y * 5)
        a = pm.Normal("a", 0.0, sigma=sigma_a, dims="groups")

        sigma_y = pm.Uniform("sigma_y", sigma_y / 100, sigma_y * 10)

        # Define the likelihood function
        likelihood = pm.Normal("likelihood", a0 + a[x_vals], sigma=sigma_y, observed=y)

        # Convert a0, a to sum-to-zero b0, b
        m = pm.Deterministic("m", a0 + a)
        b0 = pm.Deterministic("b0", at.mean(m))
        b = pm.Deterministic("b", m - b0)

        idata = pm.sample(draws=n_draws, target_accept=acceptance_rate)

        return model, idata


def hierarchical_bayesian_ancova(
    x, x_met, y, mu_x_met, mu_y, sigma_x_met, sigma_y, n_draws=1000
):
    """Models outcome resulting from categorical and metric predictors.

    The Bayesian analogue of an ANCOVA test.

    Args:
        x: The categorical predictor.
        x_met: The metric predictor.
        y: The outcome variable.
        mu_x_met: The mean of x_met.
        mu_y: The mean of y.
        sigma_x_met: The SD of x_met.
        sigma_y: The SD of y.

    Returns:
        PyMC Model and InferenceData objects.
    """
    x_vals, levels, n_levels = parse_categorical(x)

    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)

    with pm.Model(coords={"groups": levels}) as model:
        # 'a' indicates coefficients not yet obeying sum-to-zero contraint
        sigma_a = pm.Gamma("sigma_a", alpha=a_shape, beta=a_rate)
        a0 = pm.Normal("a0", mu=mu_y, sigma=sigma_y * 5)
        a = pm.Normal("a", 0.0, sigma=sigma_a, dims="groups")
        a_met = pm.Normal("a_met", mu=0, sigma=2 * sigma_y / sigma_x_met)
        # Note that in Warmhoven notebook he uses SD of residuals to set
        # lower bound on sigma_y
        sigma_y = pm.Uniform("sigma_y", sigma_y / 100, sigma_y * 10)
        mu = a0 + a[x_vals] + a_met * (x_met - mu_x_met)

        # Define the likelihood function
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma_y, observed=y)

        # Convert a0, a to sum-to-zero b0, b
        b0 = pm.Deterministic("b0", a0 + at.mean(a) + a_met * (-mu_x_met))
        b = pm.Deterministic("b", a - at.mean(a))

        # Sample from the posterior
        idata = pm.sample(draws=n_draws)

        return model, idata


def robust_bayesian_anova(x, y, mu_y, sigma_y, n_draws=1000, acceptance_rate=0.9):
    """Bayesian analogue of ANOVA using a t-distributed likelihood function.

    Args:
        x: The categorical predictor variable.
        y: The outcome variable.
        mu_y: The mean of y.
        sigma_y: The SD of y.

    Returns:
        PyMC Model and InferenceData objects.
    """
    x_vals, levels, n_levels = parse_categorical(x)

    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)

    with pm.Model(coords={"groups": levels}) as model:
        # 'a' indicates coefficients not yet obeying sum-to-zero contraint
        sigma_a = pm.Gamma("sigma_a", alpha=a_shape, beta=a_rate)
        a0 = pm.Normal("a0", mu=mu_y, sigma=sigma_y * 10)
        a = pm.Normal("a", 0.0, sigma=sigma_a, dims="groups")

        # Hyperparameters
        sigma_y_sd = pm.Gamma("sigma_y_sd", alpha=a_shape, beta=a_rate)
        sigma_y_mode = pm.Gamma("sigma_y_mode", alpha=a_shape, beta=a_rate)
        sigma_y_rate = (
            (sigma_y_mode + np.sqrt(sigma_y_mode**2 + 4 * sigma_y_sd**2))
            / 2
            * sigma_y_sd**2
        )
        sigma_y_shape = sigma_y_mode * sigma_y_rate

        sigma_y = pm.Gamma(
            "sigma", alpha=sigma_y_shape, beta=sigma_y_rate, dims="groups"
        )
        nu_minus1 = pm.Exponential("nu_minus1", 1 / 29)
        nu = pm.Deterministic("nu", nu_minus1 + 1)

        # Define the likelihood function
        likelihood = pm.Normal(
            "likelihood", a0 + a[x_vals], sigma=sigma_y[x_vals], observed=y
        )

        # Convert a0, a to sum-to-zero b0, b
        m = pm.Deterministic("m", a0 + a)
        b0 = pm.Deterministic("b0", at.mean(m))
        b = pm.Deterministic("b", m - b0)

        # Sample from the posterior. Initialization argument is necessary
        # for sampling to converge.
        idata = pm.sample(
            draws=n_draws, init="advi+adapt_diag", target_accept=acceptance_rate
        )

        return model, idata


def bayesian_two_factor_anova(x1, x2, y, n_draws=1000):
    """Bayesian analogue of two-factor ANOVA.

    Models instance of outcome resulting from two categorical predictors.

    Args:
        x1: First categorical predictor variable.
        x2: Second categorical predictor variable.
        y: The outcome variable.

    Returns:
        PyMC Model and InferenceData objects.
    """
    mu_y = y.mean()
    sigma_y = y.std()

    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)
    x1_vals, levels1, n_levels1 = parse_categorical(x1)
    x2_vals, levels2, n_levels2 = parse_categorical(x2)

    with pm.Model(coords={"factor1": levels1, "factor2": levels2}) as model:
        # To understand the reparameterization, see:
        # http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/
        a0_tilde = pm.Normal("a0_tilde", mu=0, sigma=1)
        a0 = pm.Deterministic("a0", mu_y + sigma_y * 5 * a0_tilde)

        sigma_a1 = pm.Gamma("sigma_a1", a_shape, a_rate)
        a1_tilde = pm.Normal("a1_tilde", mu=0, sigma=1, dims="factor1")
        a1 = pm.Deterministic("a1", 0.0 + sigma_a1 * a1_tilde)

        sigma_a2 = pm.Gamma("sigma_a2", a_shape, a_rate)
        a2_tilde = pm.Normal("a2_tilde", mu=0, sigma=1, dims="factor2")
        a2 = pm.Deterministic("a2", 0.0 + sigma_a2 * a2_tilde)

        sigma_a1a2 = pm.Gamma("sigma_a1a2", a_shape, a_rate)
        a1a2_tilde = pm.Normal("a1a2_tilde", mu=0, sigma=1, dims=("factor1", "factor2"))
        a1a2 = pm.Deterministic("a1a2", 0.0 + sigma_a1a2 * a1a2_tilde)

        mu = a0 + a1[x1_vals] + a2[x2_vals] + a1a2[x1_vals, x2_vals]
        sigma = pm.Uniform("sigma", sigma_y / 100, sigma_y * 10)

        # Define the likelihood function
        likelihood = pm.Normal("likelihood", mu, sigma=sigma, observed=y)

        # Sample from the posterior
        idata = pm.sample(draws=n_draws, target_accept=0.95)

        return model, idata


def two_factor_anova_convert_to_sum_to_zero(idata, x1, x2):
    """Returns coefficients that obey sum-to-zero constraint.

    Args:
        idata: InferenceData object.
        x1: First categorical predictor variable.
        x2: Second categorical predictor variable.

    Returns:
        Posterior in the form of InferenceData object.

    """
    # Extract posterior probabilities and stack your chains
    post = az.extract(idata.posterior)

    _, _, n_levels_x1 = parse_categorical(x1)
    _, _, n_levels_x2 = parse_categorical(x2)

    # Add variables
    post = post.assign(
        m=(
            ["factor1", "factor2", "sample"],
            np.zeros((n_levels_x1, n_levels_x2, len(post["sample"]))),
        )
    )
    post = post.assign(
        b1b2=(
            ["factor1", "factor2", "sample"],
            np.zeros((n_levels_x1, n_levels_x2, len(post["sample"]))),
        )
    )
    post = post.assign(b0=(["sample"], np.zeros(len(post["sample"]))))
    post = post.assign(
        b1=(["factor1", "sample"], np.zeros((n_levels_x1, len(post["sample"]))))
    )
    post = post.assign(
        b2=(["factor2", "sample"], np.zeros((n_levels_x2, len(post["sample"]))))
    )

    # Transforming the trace data to sum-to-zero values. First, calculate
    # predicted mean values based on different levels of predictors.
    for j1, j2 in np.ndindex(n_levels_x1, n_levels_x2):
        post.m[j1, j2] = (
            post["a0"] + post["a1"][j1, :] + post["a2"][j2, :] + post["a1a2"][j1, j2, :]
        )

    post["b0"] = post.m.mean(dim=["factor1", "factor2"])
    post["b1"] = post.m.mean(dim="factor2") - post.b0
    post["b2"] = post.m.mean(dim="factor1") - post.b0

    for j1, j2 in np.ndindex(n_levels_x1, n_levels_x2):
        post.b1b2[j1, j2] = post.m[j1, j2] - (post.b0 + post.b1[j1] + post.b2[j2])
    
    assert np.allclose(post.b1.sum(dim=["factor1"]), 0, atol=1e-5)
    assert np.allclose(post.b2.sum(dim=["factor2"]), 0, atol=1e-5)
    assert np.allclose(post.b1b2.sum(dim=["factor1", "factor2"]), 0, atol=1e-5)

    return post


def bayesian_oneway_rm_anova(x1, x_s, y, tune=2000, n_draws=1000):
    """Bayesian analogue of RM ANOVA.

    Models instance of outcome resulting from two categorical predictors,
    x1 and x_s (subject identifier). This model assumes each subject
    contributes only one value to each cell. Therefore, there is no
    modeling of an interaction effect (see Kruschke Ch. 20.5).

    Args:
        x1: First categorical predictor variable.
        x_s: Second categorical predictor variable - subject.
        y: The outcome variable.

    Returns:
        PyMC Model and InferenceData objects.
    """
    mu_y = y.mean()
    sigma_y = y.std()

    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)
    x1_vals, levels1, n_levels1 = parse_categorical(x1)
    x_s_vals, levels_s, n_levels_s = parse_categorical(x_s)

    with pm.Model(coords={"factor1": levels1, "factor_subj": levels_s}) as model:
        # To understand the reparameterization, see:
        # http://twiecki.github.io/blog/2017/02/08/bayesian-hierchical-non-centered/
        a0_tilde = pm.Normal("a0_tilde", mu=0, sigma=1)
        a0 = pm.Deterministic("a0", mu_y + sigma_y * 5 * a0_tilde)

        sigma_a1 = pm.Gamma("sigma_a1", a_shape, a_rate)
        # sigma_a1 = 5
        a1_tilde = pm.Normal("a1_tilde", mu=0, sigma=1, dims="factor1")
        a1 = pm.Deterministic("a1", 0.0 + sigma_a1 * a1_tilde)

        sigma_a_s = pm.Gamma("sigma_a_s", a_shape, a_rate)
        a_s_tilde = pm.Normal("a_s_tilde", mu=0, sigma=1, dims="factor_subj")
        a_s = pm.Deterministic("a_s", 0.0 + sigma_a_s * a_s_tilde)

        mu = a0 + a1[x1_vals] + a_s[x_s_vals]
        sigma = pm.Uniform("sigma", sigma_y / 100, sigma_y * 10)

        # Define the likelihood function
        likelihood = pm.Normal("likelihood", mu, sigma=sigma, observed=y)

        # Sample from the posterior
        idata = pm.sample(draws=n_draws, target_accept=0.95)

        return model, idata


def oneway_rm_anova_convert_to_sum_to_zero(idata, x1, x_s):
    """Returns coefficients that obey sum-to-zero constraint.

    Args:
        idata: InferenceData object.
        x1: First categorical predictor variable.
        x2: Second categorical predictor variable.

    Returns:
        Posterior in the form of InferenceData object.

    """
    # Extract posterior probabilities and stack your chains
    # post = az.extract(idata.posterior)
    post = az.extract_dataset(idata.posterior)

    _, _, n_levels_x1 = parse_categorical(x1)
    _, _, n_levels_x_s = parse_categorical(x_s)

    # Add variables
    post = post.assign(
        m=(
            ["factor1", "factor_subj", "sample"],
            np.zeros((n_levels_x1, n_levels_x_s, len(post["sample"]))),
        )
    )
    post = post.assign(b0=(["sample"], np.zeros(len(post["sample"]))))
    post = post.assign(
        b1=(["factor1", "sample"], np.zeros((n_levels_x1, len(post["sample"]))))
    )
    post = post.assign(
        b_s=(["factor_subj", "sample"], np.zeros((n_levels_x_s, len(post["sample"]))))
    )

    # Transforming the trace data to sum-to-zero values. First, calculate
    # predicted mean values based on different levels of predictors.
    for j1, j2 in np.ndindex(n_levels_x1, n_levels_x_s):
        post.m[j1, j2] = post["a0"] + post["a1"][j1, :] + post["a_s"][j2, :]

    post["b0"] = post.m.mean(dim=["factor1", "factor_subj"])
    post["b1"] = post.m.mean(dim="factor_subj") - post.b0
    post["b_s"] = post.m.mean(dim="factor1") - post.b0
    
    assert np.allclose(post.b1.sum(dim=["factor1"]), 0, atol=1e-5)
    assert np.allclose(post.b_s.sum(dim=["factor_subj"]), 0, atol=1e-5)
    
    return post


def bayesian_mixed_model_anova(
    between_subj_var, within_subj_var, subj_id, y, n_samples=1000
):
    """Performs Bayesian analogue of mixed model (split-plot) ANOVA.

    Models instance of outcome resulting from both between- and within-subjects
    factors. Outcome is measured several times from each observational unit (i.e.,
    repeated measures).

    Args:
        between_subj_var: The between-subjects variable.
        withing_subj_var: The within-subjects variable.
        subj_id: The subj ID variable.
        y: The outcome variable.

    Returns:
        PyMC Model and InferenceData objects.
    """
    # Statistical model: Split-plot design after Kruschke Ch. 20
    # Between-subjects factor (i.e., group)
    x_between, levels_x_between, num_levels_x_between = parse_categorical(
        between_subj_var
    )

    # Within-subjects factor (i.e., target set)
    x_within, levels_x_within, num_levels_x_within = parse_categorical(within_subj_var)

    # Individual subjects
    x_subj, levels_x_subj, num_levels_x_subj = parse_categorical(subj_id)

    # Dependent variable
    mu_y = y.mean()
    sigma_y = y.std()

    a_shape, a_rate = gamma_shape_rate_from_mode_sd(sigma_y / 2, 2 * sigma_y)

    with pm.Model(
        coords={
            "between_subj": levels_x_between,
            "within_subj": levels_x_within,
            "subj": levels_x_subj,
        }
    ) as model:
        # Baseline value
        a0 = pm.Normal("a0", mu=mu_y, sigma=sigma_y * 5)

        # Deflection from baseline for between subjects factor
        sigma_B = pm.Gamma("sigma_B", a_shape, a_rate)
        aB = pm.Normal("aB", mu=0.0, sigma=sigma_B, dims="between_subj")

        # Deflection from baseline for within subjects factor
        sigma_W = pm.Gamma("sigma_W", a_shape, a_rate)
        aW = pm.Normal("aW", mu=0.0, sigma=sigma_W, dims="within_subj")

        # Deflection from baseline for combination of between and within subjects factors
        sigma_BxW = pm.Gamma("sigma_BxW", a_shape, a_rate)
        aBxW = pm.Normal(
            "aBxW", mu=0.0, sigma=sigma_BxW, dims=("between_subj", "within_subj")
        )

        # Deflection from baseline for individual subjects
        sigma_S = pm.Gamma("sigma_S", a_shape, a_rate)
        aS = pm.Normal("aS", mu=0.0, sigma=sigma_S, dims="subj")

        mu = a0 + aB[x_between] + aW[x_within] + aBxW[x_between, x_within] + aS[x_subj]
        sigma = pm.Uniform("sigma", lower=sigma_y / 100, upper=sigma_y * 10)

        # Define likelihood
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)

        # Sample from the posterior
        idata = pm.sample(draws=n_samples, tune=2000, target_accept=0.95)

    return model, idata


def unpack_posterior_vars(posterior):
    """Unpacks posterior variables from xarray structure.

    Intended for use with bayesian_mixed_model_anova.

    Args:
        posterior: Posterior variables from InferenceData object.

    Returns:
        Posterior variables.
    """
    a0 = posterior["a0"]
    aB = posterior["aB"]
    aW = posterior["aW"]
    aBxW = posterior["aBxW"]
    aS = posterior["aS"]
    sigma = posterior["sigma"]

    return a0, aB, aW, aBxW, aS, sigma


def create_masked_array(
    a0, aB, aW, aBxW, aS, posterior, between_subj_var, within_subj_var, subj_id
):
    """Creates a masked array with all cell values from the posterior."""
    # Between-subjects factor
    x_between, levels_x_between, num_levels_x_between = parse_categorical(
        between_subj_var
    )

    # Within-subjects factor
    x_within, levels_x_within, num_levels_x_within = parse_categorical(within_subj_var)

    # Individual subjects
    x_subj, levels_x_subj, num_levels_x_subj = parse_categorical(subj_id)

    # Initialize the array with zeros
    posterior = posterior.assign(
        m_SxBxW=(
            ["subj", "between_subj", "within_subj", "sample"],
            np.zeros(
                (
                    num_levels_x_subj,
                    num_levels_x_between,
                    num_levels_x_within,
                    len(posterior["sample"]),
                )
            ),
        )
    )

    # Fill the arrray
    for k, i, j in zip(x_subj, x_between, x_within):
        posterior.m_SxBxW[k, i, j, :] = (
            a0 + aB[i, :] + aW[j, :] + aBxW[i, j, :] + aS[k, :]
        )

    # Convert to masked array that masks value '0'.
    posterior = posterior.assign(
        m_SxBxW=(
            ["subj", "between_subj", "within_subj", "sample"],
            (ma.masked_equal(posterior.m_SxBxW, 0)),
        )
    )

    m_SxBxW = posterior.m_SxBxW

    return m_SxBxW


def calc_marginal_means(m_SxBxW):
    """Calculate the marginalized means using the masked array."""

    # Mean for subject S across levels of W, within the level of B
    m_S = m_SxBxW.mean(dim="within_subj")

    # Mean for treatment combination BxW, across subjects S
    m_BxW = m_SxBxW.mean(dim="subj")

    # Mean for level B, across W and S
    m_B = m_BxW.mean(dim=["within_subj"])

    # Mean for level W, across B and S
    m_W = m_BxW.mean(dim=["between_subj"])

    return m_S, m_BxW, m_B, m_W


def convert_to_sum_to_zero(idata, between_subj_var, within_subj_var, subj_id):
    """Returns coefficients that obey sum-to-zero constraint.

    Args:
        idata: InferenceData object.
        between_subj_var: Between-subjects predictor variable.
        within_subj_var: Within-subjects predictor variable.
        subj_id: Subject ID variable.

    Returns:
        Posterior variables.
    """

    posterior = az.extract(idata.posterior)
    a0, aB, aW, aBxW, aS, sigma = unpack_posterior_vars(posterior)
    m_SxBxW = create_masked_array(
        a0, aB, aW, aBxW, aS, posterior, between_subj_var, within_subj_var, subj_id
    )
    m_S, m_BxW, m_B, m_W = calc_marginal_means(m_SxBxW)

    # Equation 20.3
    m = m_BxW.mean(dim=["between_subj", "within_subj"])
    posterior = posterior.assign(b0=m)

    # Equation 20.4
    posterior = posterior.assign(bB=m_B - m)

    # Equation 20.5
    posterior = posterior.assign(bW=m_W - m)

    # Equation 20.6
    posterior = posterior.assign(bBxW=m_BxW - m_B - m_W + m)

    # Equation 20.7 (removing between_subj dimension)
    posterior = posterior.assign(
        bS=(["subj", "sample"], (m_S - m_B).mean(dim="between_subj").data)
    )
    
    assert np.allclose(posterior.bB.sum(dim=["between_subj"]), 0, atol=1e-5)
    assert np.allclose(posterior.bW.sum(dim=["within_subj"]), 0, atol=1e-5)
    assert np.allclose(posterior.bBxW.sum(dim=["between_subj", "within_subj"]), 0, atol=1e-5)
    assert np.allclose(posterior.bS.sum(dim=["subj"]), 0, atol=1e-5)

    return posterior


def bayesian_logreg_cat_predictors(X, y, n_draws=1000):
    # Standardize the predictor variable(s)
    zX, mu_X, sigma_X = standardize(X)

    with pm.Model(coords={"predictors": X.columns.values}) as model:
        # Set  priors
        zbeta0 = pm.Normal("zbeta0", mu=0, sigma=2)
        zbetaj = pm.Normal("zbetaj", mu=0, sigma=2, dims="predictors")

        p = pm.invlogit(zbeta0 + pm.math.dot(zX, zbetaj))

        # Define likelihood function
        likelihood = pm.Bernoulli("likelihood", p, observed=y)

        # Transform parameters to original scale
        beta0 = pm.Deterministic(
            "beta0", (zbeta0 - pm.math.sum(zbetaj * mu_X / sigma_X))
        )
        betaj = pm.Deterministic("betaj", zbetaj / sigma_X, dims="predictors")

        # Sample from the posterior
        idata = pm.sample(draws=n_draws)

        return model, idata


def bayesian_logreg_subj_intercepts(subj, X, y, n_draws=1000):
    # Factorize subj IDs and treatment variable
    subj_idx, subj_levels, n_subj = parse_categorical(subj)
    treatment_idx, treatment_levels, n_treatment = parse_categorical(X)

    with pm.Model(coords={"subj": subj_levels, "treatment": treatment_levels}) as model:
        # Set priors
        a = pm.Normal("a", 0.0, 1.5, dims="subj")
        b = pm.Normal("b", 0.0, 0.5, dims="treatment")

        p = pm.Deterministic("p", pm.math.invlogit(a[subj_idx] + b[treatment_idx]))

        # Define likelihood function (in this case, observations are 0 or 1)
        likelihood = pm.Binomial("likelihood", 1, p, observed=y)

        # Draw samples from the posterior
        idata = pm.sample(draws=n_draws)

        return model, idata
