[![DOI](https://zenodo.org/badge/553182204.svg)](https://zenodo.org/badge/latestdoi/553182204)

# Bayesian Statistics Toolbox (BST) 
---
BST is a library of functions for running sophisticated Bayesian analyses in a simple, straight forward manner, and all in Python. 


## Documentation
---
The documentation for `BST` is hosted [here](https://hyosubkim.github.io/bayesian-statistics-toolbox/) on a GitHub Pages site.


## Installation 
---
See the [installation instructions](https://hyosubkim.github.io/bayesian-statistics-toolbox/getting-started/) in the [documentation](https://hyosubkim.github.io/bayesian-statistics-toolbox/) for detailed instructions on how to install BST. 


## Citing BST
---
If you use BST in your work, please cite use one of the following:

***APA format:***
- Kim, H. E. (2022). Bayesian Statistics Toolbox (BST) (Version 1.0) [Computer software]. https://doi.org/10.5281/zenodo.7339667

***BibTeX format:***
```
@software{Kim_Bayesian_Statistics_Toolbox_2022,
author = {Kim, Hyosub E.},
doi = {10.5281/zenodo.7339667},
month = {11},
title = {{Bayesian Statistics Toolbox (BST)}},
url = {https://github.com/hyosubkim/bayesian-statistics-toolbox},
version = {1.0},
year = {2022}
}
```

## License
---
This work is distributed for free under a [MIT license](https://github.com/hyosubkim/bayesian-statistics-toolbox/blob/main/LICENSE). 

## Acknowledgments
---
Thank you to the following people for generously sharing their work and knowledge:
- [John Kruschke](https://jkkweb.sitehost.iu.edu/)
- [Richard McElreath](https://xcelab.net/rm/)
- [Jordi Warmenhoven](https://github.com/JWarmenhoven)
- [PyMC developers](https://github.com/pymc-devs/pymc)
- [ArviZ developers](https://www.arviz.org/en/latest/our_team.html)

I have tried my best to give credit for your work wherever appropriate, but please let me know if you would like your work attributed in a different manner. 

## Contributors
---
See the AUTHORS.md file for a regularly updated list of contributors. 


BST provides you with the tools for utilizing and exploring Bayesian statistics in your own research projects right away. I've included example use cases for almost every model provided (in the `examples` directory), so you can see for yourself what a sensible Bayesian data analysis pipeline looks like. The example notebooks are mostly adaptations of [Jordi Warmenhoven's Python/PyMC3 port](https://github.com/JWarmenhoven/DBDA-python) of John Kruschke's excellent textbook ["Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan"](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0). BST is in large part updating Jordi Warmenhoven's original PyMC3 versions of the Kruschke models to [PyMC 4.0](https://www.pymc.io/welcome.html) and wrapping them into tidy functions so that they are easily re-usable. The benefits of using BST in terms of time and convenience will be most noticeable when utilizing hierarchical (multi-level) models, including the ANOVA-like ones. 

In addition, BST takes care of some of the more finicky steps involved in Bayesian statistical modeling with embedded functions for things like standardizing/unstandardizing variables for more efficient MCMC sampling, parsing categorical variables for easier indexing, and implementing sum-to-zero constraints in ANOVA-like models. These are the sorts of implementational details that can add time (and frustration) when creating an analysis pipeline and discourage otherwise interested scientists from using Bayesian statistics. I hope BST removes those obstacles as well.  
 
## Why is this useful?
By wrapping model definitions of Bayesian generalized linear models into convenient functions, BST makes it easier to run Bayesian analyses that are analogous to some of the most commonly used frequentist tests in the behavioral and neural sciences (think t-tests, ANOVAs, regression). I originally started working on BST because I wanted to be able to utilize Bayesian statistics in my own research without having to code each model up from scratch, as that itself can be a barrier (especially when there is the temptation to fall back into frequentist habits using one-liners from `statsmodels`). Soon, I realized that this project may actually lower the bar to entry into the beautiful world of Bayesian statistics for others as well, and so here we are. 

BST is also useful if you're going through the DBDA textbook and want to see how to implement the models in PyMC 4.0. However, this project is still a work in progress and does not cover all of the models in DBDA. For a much more complete Python-ic implementation (in PyMC3), see Jordi Warmenhoven's [repo](https://github.com/JWarmenhoven/DBDA-python).

Please note that the models all utilize fairly uninformative, diffuse priors, which are, for the most part, the exact same ones used in the Kruschke text. An easy way to modify the priors, or any part of the model, for that matter, is to print the function (add two question marks after the function name) and copy it over to your editor. 

## Dependencies
Some of the main libraries used in this project:

- [ArviZ](https://arviz-devs.github.io/arviz/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [PyMC](https://www.pymc.io/welcome.html)
- [seaborn](https://seaborn.pydata.org/)
- [Xarray](https://docs.xarray.dev/en/stable/)


## How do I get started?
*(Recommended)*   
If you're running Mac OSX, and want to ensure you can run everything right out of the box, after cloning this repo you can create a virtual environment with all of the necessary dependencies. To do that, make sure you're in the root directory of this repository (i.e., `bayesian-statistics-toolbox`) and type the following conda command in the Terminal (I strongly recommend [Anaconda](https://www.anaconda.com/) to install Python and the conda utility on your computer):
```
conda env create --name bayes_toolbox --file environment.yml
```
You can name your environment whatever you like, it doesn't have to be "bayes_toolbox". 

**To access all of the functions, pip install the package from the Terminal. Again, make sure you are in the root directory for this repository and then at the prompt type in:**
```
pip install -e .
```

Once installed locally, you can access it from any directory. When you import your packages, add the following line:

```python
import src.bayesian_stats as bst
```

If you're on a different OS and want to replicate this environment, read the "Export your environment" section of this [page](https://goodresearch.dev/setup.html). 

To access the correct kernel from a Jupyter notebook, you must manually add the kernel for your new virtual environment "bayes_toolbox" (or whatever you named it). To do so, you first need to install [ipykernel](https://github.com/ipython/ipykernel):
```
pip install --user ipykernel
```

Next, add your virtual environment to Jupyter:
```
python -m ipykernel install --user --name=MYENV
```
Use whatever you named your virtual environment in place of `MYENV`. That should be all that's necessary in order to choose your new virtual environment as a kernel from your Jupyter notebook. For more details, read this [page](https://janakiev.com/blog/jupyter-virtual-envs/). 


## How do I learn to use BST?
The `BEST`  notebook (short for "Bayesian Estimation Supersedes the t-Test", a famous 2013 [article](https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf) by John Kruschke) in `examples` is a good place to see how BST can be used to make Bayesian analyses more convenient. I've adapted the [notebook](https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html) of the same name from the PyMC developers to show how the model building and MCMC sampling are all embedded in a single function now. You can see similar workflows for other model types in all of the other example notebooks, which track several of the chapters from "Doing Bayesian Data Analysis" and is modeled off of Jordi Warmenhoven's [repo](https://github.com/JWarmenhoven/DBDA-python).

## Example syntax
Now, if you want to run a fairly sophisticated multi-level (hierarchical) linear regression model in which you are modeling individual as well as group-level slope and intercept parameters, it looks like

```python
# Call your BST function and return the PyMC model and InferenceData objects
model, idata = bst.hierarchical_regression(df["x"], df["y"], df["subj"], acceptance_rate=0.95)
```
Before, this would have taken *many* more lines of code. 

## Where can I get more help?
If you're just starting off with Bayesian statistics, the two best introductory textbooks I know of are ["Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan"](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0) by John Kruschke and ["Statistical Rethinking"](https://xcelab.net/rm/statistical-rethinking/) by Richard McElreath. If you already have some background in Bayesian inference but are new to Python and/or PyMC, the PyMC developers have some great example notebooks [here](https://www.pymc.io/projects/examples/en/latest/gallery.html).

## How to contribute to BST

Thank you for your interest in contributing to BST! Go [here](https://github.com/hyosubkim/bayesian-statistics-toolbox/blob/main/CONTRIBUTING.md) for some contributing guidelines.  

## Models that are currently included and validated (frequentist analogue in parentheses)
- Comparison of two groups (independent samples t-test)
- Comparison of single or paired samples (paired t-test)
- Simple linear regression
- Multiple regression
- Multi-level (hierarchical) linear regression for modeling group- and individual-specific parameters
- Hierarchical (multi-level) model of metric outcome with single categorical predictor (one-way ANOVA)
- Hierarchical (multi-level) model of metric outcome with single categorical and single metric predictors (ANCOVA)
- Hierarchical (multi-level) model of metric outcome with two categorical predictors (two-way ANOVA)
- Hierarchical (multi-level) model of metric outcome with multiple categorical predictors and repeated measures (mixed-model ANOVA)
- Logistic regression models incorporating categorical or metric predictors

## Are there any plans to further develop BST?
Yes, there will be much more coming soon! I plan to add the following in the near future:

- Bayes Factors
- Bayesian meta-analysis 
- Bayesian spectral analysis
- Model selection
- Incorporating prior and posterior predictive checks into all of the examples (currently, only logistic regression notebook has example of prior predictive check)
- Incorporating maximum entropy priors 

## Other related Python projects 
For a more weapons-grade Bayesian statistical modeling interface, check out:
- [Bambi](https://github.com/bambinos/bambi): BAyesian Model-Building Interface (BAMBI) in Python.

## Citing BST
If you use BST and would like to cite, please use one of the following:

***APA format:***
- Kim, H. E. (2022). Bayesian Statistics Toolbox (BST) (Version 1.0) [Computer software]. https://doi.org/10.5281/zenodo.7339667

***BibTeX format:***
```
@software{Kim_Bayesian_Statistics_Toolbox_2022,
author = {Kim, Hyosub E.},
doi = {10.5281/zenodo.7339667},
month = {11},
title = {{Bayesian Statistics Toolbox (BST)}},
url = {https://github.com/hyosubkim/bayesian-statistics-toolbox},
version = {1.0},
year = {2022}
}
```

## License
This work is distributed under a [MIT license](https://github.com/hyosubkim/bayesian-statistics-toolbox/blob/main/LICENSE). 


