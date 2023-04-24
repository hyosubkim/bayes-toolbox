# bayes-toolbox 

[![DOI](https://zenodo.org/badge/553182204.svg)](https://zenodo.org/badge/latestdoi/553182204)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`bayes-toolbox` (aka, Bayesian Statistics Toolbox [BST]) is a Python package for running sophisticated Bayesian analyses in a simple, straight forward manner. 

## Why is this useful?

### Research
`bayes-toolbox` provides you with the tools for utilizing and exploring Bayesian statistics in your own research projects right away. By wrapping model definitions of Bayesian generalized linear models into convenient functions, `bayes-toolbox` makes it easy to run Bayesian analyses that are analogous to some of the most commonly used frequentist tests in the behavioral and neural sciences (think t-tests, ANOVAs, regression). All of the models (tests) are executable with one-liners, thereby making them ideal for use in an open, replicable (and Bayesian) workflow (watch this [PyMCon talk](https://www.youtube.com/watch?v=ElfToZ9EBpM) to learn more). **Example use cases and tests for nearly every model are provided in the [`examples`](https://github.com/hyosubkim/bayes-toolbox/tree/main/examples) directory, so you can see what a sensible Bayesian data analysis pipeline looks like.** (Many of the example notebooks are adaptations of [Jordi Warmenhoven's Python/PyMC3 port](https://github.com/JWarmenhoven/DBDA-python) of John Kruschke's excellent textbook ["Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan (DBDA)"](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0).

The benefits of using `bayes-toolbox` in terms of time and convenience will be most noticeable when utilizing hierarchical (multi-level) models, including the ANOVA-like ones. This is because `bayes-toolbox` takes care of the more finicky steps involved in Bayesian statistical modeling with embedded functions for things like standardizing/unstandardizing variables for more efficient MCMC sampling, parsing categorical variables for easier indexing, and implementing sum-to-zero constraints in ANOVA-like models. These are the sorts of implementational details that can add time (and frustration) when creating an analysis pipeline and discourage otherwise interested scientists from using Bayesian statistics. `bayes-toolbox` now removes those obstacles. 

The package is actively being developed in a public [GitHub repository](https://github.com/hyosubkim/bayes-toolbox), and we always welcome new contributors! No contribution is too small. If you have any issues with the code, suggestions on how to improve it, or have requests for additional content, whether new features or tutorials, please [open an issue on Github](https://github.com/hyosubkim/bayes-toolbox/issues). 

### Education
`bayes-toolbox` will be very useful if you are going through [Doing Bayesian Data Analysis](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0) and want to learn how to implement the models in Python/PyMC. 

## Look before you leap
Please note that the models in `bayes-toolbox` all utilize fairly uninformative, diffuse priors, which are, for the most part, the exact same ones used in the Kruschke text. Also, keep in mind that even though `bayes-toolbox` offers a streamlined interface for performing common statistical tests, the assumptions John Kruschke and we, the developers, make in these models may not be the ones you want to make for your particular question. Therefore, it's a good idea to go through the notebooks in the [`examples`](https://github.com/hyosubkim/bayes-toolbox/tree/main/examples) to make sure the model is appropriate for your applications. Part of the beauty of Bayesian modeling is its flexibility, so if you want to change priors/hyperpriors/etc., feel free to use `bayes-toolbox` as model scaffolding for a new bespoke model fit to your purpose. And consider making it a [contribution](https://hyosubkim.github.io/bayes-toolbox/how-to-contribute/)! 

## Dependencies
Some of the main libraries used in this project:

- [ArviZ](https://arviz-devs.github.io/arviz/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [PyMC](https://www.pymc.io/welcome.html)
- [seaborn](https://seaborn.pydata.org/)
- [Xarray](https://docs.xarray.dev/en/stable/)

---
**NOTE**      
Use the links in the navigation bar to the left, the search bar in the upper left, or the content pages below to get started!  

---
[Getting Started](https://hyosubkim.github.io/bayes-toolbox/getting-started/)  
[Tutorials](https://hyosubkim.github.io/bayes-toolbox/tutorials/)  
[Future Plans](https://hyosubkim.github.io/bayes-toolbox/future-plans/)     
[How to Contribute](https://hyosubkim.github.io/bayes-toolbox/how-to-contribute/)

## Bayesian models that are currently included and validated (frequentist analogue in parentheses)
- Comparison of two groups *(independent samples t-test)*
- Comparison of single or paired samples *(paired t-test)*
- Simple linear regression
- Multiple regression
- Multi-level (hierarchical) linear regression for modeling group- and individual-specific parameters
- Hierarchical (multi-level) model of metric outcome with single categorical predictor *(one-way ANOVA)*
- Hierarchical (multi-level) model of metric outcome with single categorical and single metric predictors *(ANCOVA)*
- Hierarchical (multi-level) model of metric outcome with two categorical predictors *(two-way ANOVA)*
- Hierarchical (multi-level) model of metric outcome with multiple categorical predictors and repeated measures *(mixed-model ANOVA)*
- Logistic regression models incorporating categorical or metric predictors
- Meta-analysis of binary outcomes using *random effects* model

## Other related Python projects 
For a more weapons-grade Bayesian statistical modeling interface, check out:  

- [Bambi](https://github.com/bambinos/bambi): BAyesian Model-Building Interface (BAMBI) in Python.

While `Bambi` requires model formulas, `bayes-toolbox` instead requires calling the function associated with a particular test. 

## Citing BST
If you use BST and would like to cite, please use one of the following:

***APA format:***  
    Kim, H. E. (2023). bayes-toolbox (Version 0.0.1) [Computer software]. https://doi.org/10.5281/zenodo.7339667
    
***BibTeX format:***
```
@software{Kim_bayes-toolbox_2023,
author = {Kim, Hyosub E.},
doi = {10.5281/zenodo.7339667},
month = apr,
title = {{bayes-toolbox}},
url = {https://github.com/hyosubkim/bayes-toolbox},
version = {0.0.1},
year = {2023}
}
```

## License
This work is distributed under a [MIT license](https://github.com/hyosubkim/bayesian-statistics-toolbox/blob/main/LICENSE). 

## Acknowledgments
Thank you to the following people for generously sharing their work and knowledge:  
- [John Kruschke](https://jkkweb.sitehost.iu.edu/)  
- [Richard McElreath](https://xcelab.net/rm/)  
- [Jordi Warmenhoven](https://github.com/JWarmenhoven) - *This project grew out of updating Jordi's great Python/PyMC 3.0. port of the Kruschke textbook.*   
- [PyMC developers](https://github.com/pymc-devs/pymc)  
- [ArviZ developers](https://www.arviz.org/en/latest/our_team.html)  


