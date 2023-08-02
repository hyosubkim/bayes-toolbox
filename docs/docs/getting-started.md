# How do I get started?

## Installation

### Recommended (pip)
We recommend installing the latest stable version of `bayes_toolbox` with pip (from the Terminal):
```
pip install bayes_toolbox
```

### From source: Cloning and building
The latest development version of `bayes-toolbox` can be cloned from GitHub using `git`:
```
git clone https://github.com/hyosubkim/bayes-toolbox.git
```

To build and install the project (from the root directory, i.e., inside the cloned `bayes-toolbox` directory):
```
python3 -m pip install -e .
```

## Dependencies
`bayes_toolbox` has the following dependencies (all of which should be automatically imported with a pip installation):  
- aesara  
- arviz  
- numpy  
- pandas  
- pymc  


### Virtual environment
You can create a virtual environment with all of the necessary dependencies. If you've cloned the `bayes-toolbox` repository, make sure you're in the root directory of the cloned repository (i.e., `bayes-toolbox`) and type the following conda command in the Terminal ([Anaconda](https://www.anaconda.com/) is strongly recommended for installing Python and the conda utility on your computer):
```
conda env create --name bayes_toolbox --file environment.yml
```
Instead of `bayes_toolbox`, you may wish to give your environment a different name. 

If you're not using MacOSX and want to replicate this environment, read the "Export your environment" section of this [page](https://goodresearch.dev/setup.html). 

After cloning and installing `bayes-toolbox` locally, you can access it from any directory. 

### Helpful hints
If you've created the `bayes_toolbox` virtual environment and want to access the correct kernel from a Jupyter notebook, you must manually add the kernel for your new virtual environment "bayes_toolbox" (or whatever you named it). To do so, you first need to install [ipykernel](https://github.com/ipython/ipykernel):
```
pip install --user ipykernel
```

Next, add your virtual environment to Jupyter:
```
python -m ipykernel install --user --name=MYENV
```
Use whatever you named your virtual environment in place of `MYENV`. That should be all that's necessary in order to choose your new virtual environment as a kernel from your Jupyter notebook. For more details, read this [page](https://janakiev.com/blog/jupyter-virtual-envs/). 

## How do I learn to use bayes-toolbox?
The `BEST`  notebook (short for "Bayesian Estimation Supersedes the t-Test", a famous 2013 [article](https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf) by John Kruschke) in the `examples` [directory](https://github.com/hyosubkim/bayesian-statistics-toolbox/tree/main/examples) is a good place to see how `bayes-toolbox` can be used to make implementing Bayesian analyses easy. I've adapted the [notebook](https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html) of the same name from the PyMC developers to show how the model building and MCMC sampling are all embedded in a single function now. You can see similar workflows for other model types in all of the other example notebooks, which track several of the chapters from "Doing Bayesian Data Analysis" and is modeled off of Jordi Warmenhoven's [repo](https://github.com/JWarmenhoven/DBDA-python).

## Example syntax
Following imports of the most common Python packages for data analysis and Bayesian statistics, import `bayes_toolbox`. 

```python
# Usual imports 
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr

# Import the bayes-toolbox package 
import bayes_toolbox.glm as bg
```

Import the data you want to model (the following example can be found in the `examples` subdirectory). So far, these are all standard steps and not specific to `bayes-toolbox`. 

```python
# Import data (from 'examples' subdirectory) into pandas data frame 
df = pd.read_csv("data/HierLinRegressData.csv")
df.Subj = df_HRegr.Subj.astype("category")
df.Subj = df_HRegr.Subj.cat.as_ordered()
```

**Now, with `bayes-toolbox`, if you want to run a fairly sophisticated multi-level (hierarchical) linear regression model in which you are modeling individual as well as group-level slope and intercept parameters, simply call the appropriate function:**

```python
# Call your bayes-toolbox function and return the PyMC model and InferenceData objects
model, idata = bg.hierarchical_regression(
df["X"], df["Y"], df["Subj"], acceptance_rate=0.95
)
```

Before, this exact same analysis would have taken *many* more lines of code:

```python
# Standardize variables
zx, mu_x, sigma_x = standardize(df["X"])
zy, mu_y, sigma_y = standardize(df["Y")

# Convert subject variable to categorical dtype if it is not already
subj_idx, subj_levels, n_subj = parse_categorical(df["Subj"])

# Define your statistical model
with pm.Model(coords={"subj": subj_levels}) as model:
    # Hyperpriors
    zbeta0 = pm.Normal("zbeta0", mu=0, tau=1 / 10**2)
    zbeta1 = pm.Normal("zbeta1", mu=0, tau=1 / 10**2)
    zsigma0 = pm.Uniform("zsigma0", 10**-3, 10**3)
    zsigma1 = pm.Uniform("zsigma1", 10**-3, 10**3)

    # Priors for individual subject parameters
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
```
