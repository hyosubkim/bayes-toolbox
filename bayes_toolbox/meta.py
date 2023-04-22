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

"""
A collection of Bayesian statistical models and associated utility functions.
"""

import aesara.tensor as at
import arviz as az
import numpy as np
import numpy.ma as ma
import pandas as pd
import pymc as pm


def meta_binary_outcome(z_t_obs, n_t_obs, z_c_obs, n_c_obs, study, n_draws=1000):
    """Fits multi-level meta-analysis model of binary outcomes.

    See meta-analysis-two-proportions.ipynb in examples for usage.

    Args:
        z_t_obs: number of occurrences in treatment group
        n_t_obs: number of opportunities/participants in treatment gruops
        z_c_obs: number of occurrences in control group
        n_c_obs: number of opportunities/participants in control group
        study: list of studies included in analysis

    Returns:
        PyMC model and InferenceData objects.
    """

    with pm.Model(coords={"study": study}) as model:
        # Hyper-priors
        mu_rho = pm.Normal("mu_rho", mu=0, sigma=10)
        sigma_rho = pm.Gamma("sigma_rho", alpha=1.64, beta=0.64)  # mode=1, sd=2

        omega_theta_c = pm.Beta("omega_theta_c", alpha=1.01, beta=1.01)
        kappa_minus_two_theta_c = pm.Gamma(
            "kappa_minus_two_theta_c", alpha=2.618, beta=0.162
        )  # mode=10, sd=10
        kappa_theta_c = pm.Deterministic("kappa_theta_c", kappa_minus_two_theta_c + 2)

        # Priors
        rho = pm.Normal("rho", mu=mu_rho, sigma=sigma_rho, dims="study")
        theta_c = pm.Beta(
            "theta_c",
            alpha=omega_theta_c * (kappa_theta_c - 2) + 1,
            beta=(1 - omega_theta_c) * (kappa_theta_c - 2) + 1,
            dims="study",
        )
        theta_t = pm.Deterministic(
            "theta_t", pm.invlogit(rho + pm.logit(theta_c))
        )  # ilogit is logistic

        # Likelihood
        z_t = pm.Binomial("z_t", n_t_obs, theta_t, observed=z_t_obs)
        z_c = pm.Binomial("z_c", n_c_obs, theta_c, observed=z_c_obs)

        # Sample from the posterior
        idata = pm.sample(draws=n_draws, target_accept=0.90)

        return model, idata
