# Welcome to the Bayesian Statistics Toolbox (BST)

[![DOI](https://zenodo.org/badge/553182204.svg)](https://zenodo.org/badge/latestdoi/553182204)

# Bayesian Statistics Toolbox (BST) 

BST is a library of functions for running sophisticated Bayesian analyses in a simple, straight forward manner, and all in Python. The package is actively being developed in a public [GitHub repository](https://github.com/hyosubkim/bayesian-statistics-toolbox). 

BST provides you with the tools for utilizing and exploring Bayesian statistics in your own research projects right away. I've included example use cases for almost every model provided (in the [`examples` directory](https://github.com/hyosubkim/bayesian-statistics-toolbox/tree/main/examples), so you can see for yourself what a sensible Bayesian data analysis pipeline looks like. The example notebooks are mostly adaptations of [Jordi Warmenhoven's Python/PyMC3 port](https://github.com/JWarmenhoven/DBDA-python) of John Kruschke's excellent textbook ["Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan"](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0). BST is in large part updating Jordi Warmenhoven's original PyMC3 versions of the Kruschke models to [PyMC 4.0 (soon to be updated to PyMC 5)](https://www.pymc.io/welcome.html) and wrapping them into tidy functions so that they are easily re-usable. The benefits of using BST in terms of time and convenience will be most noticeable when utilizing hierarchical (multi-level) models, including the ANOVA-like ones. 

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

---
**NOTE**      
Use the links in the navigation bar to the left, the search bar in the upper left, or the content pages below to get started!  

---
[Getting Started](https://hyosubkim.github.io/bayesian-statistics-toolbox/getting-started/)  
[Tutorials](https://hyosubkim.github.io/bayesian-statistics-toolbox/tutorials/)  
[Future Plans](https://hyosubkim.github.io/bayesian-statistics-toolbox/future-plans/)     
[How to Contribute](https://hyosubkim.github.io/bayesian-statistics-toolbox/how-to-contribute/)

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

## Other related Python projects 
For a more weapons-grade Bayesian statistical modeling interface, check out:  

- [Bambi](https://github.com/bambinos/bambi): BAyesian Model-Building Interface (BAMBI) in Python.

## Citing BST
If you use BST and would like to cite, please use one of the following:

***APA format:***  
    Kim, H. E. (2022). Bayesian Statistics Toolbox (BST) (Version 1.0) [Computer software]. https://doi.org/10.5281/zenodo.7339667

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

## Acknowledgments
Thank you to the following people for generously sharing their work and knowledge:  
- [John Kruschke](https://jkkweb.sitehost.iu.edu/)  
- [Richard McElreath](https://xcelab.net/rm/)  
- [Jordi Warmenhoven](https://github.com/JWarmenhoven)  
- [PyMC developers](https://github.com/pymc-devs/pymc)  
- [ArviZ developers](https://www.arviz.org/en/latest/our_team.html)  

I have tried my best to give credit for your work wherever appropriate, but please let me know if you would like your work attributed in a different manner. 

