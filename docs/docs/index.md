# bayes-toolbox 

[![DOI](https://zenodo.org/badge/553182204.svg)](https://zenodo.org/badge/latestdoi/553182204)
[![status](https://joss.theoj.org/papers/1b7b8068a329b547e28d00da0ad790b2/status.svg)](https://joss.theoj.org/papers/1b7b8068a329b547e28d00da0ad790b2)
![coverage](coverage.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`bayes-toolbox` (aka, Bayesian Statistics Toolbox [BST]) is a Python package for running sophisticated Bayesian analyses in a simple, straight forward manner. 

## Statement of Need 

### Research
`bayes-toolbox` provides you with the tools for utilizing and exploring Bayesian statistics in your own research projects right away. By wrapping model definitions of Bayesian generalized linear models into convenient functions, `bayes-toolbox` makes it easy to run Bayesian analyses that are analogous to some of the most commonly used frequentist tests in the behavioral and neural sciences (think t-tests, ANOVAs, regression). Right now, Python users can choose between several packages that allow for one-liners to be called in order to run classical/frequentist tests (e.g., [Pingouin](https://pingouin-stats.org/build/html/index.html#), [SciPy](https://scipy.org/), [pandas](https://pandas.pydata.org/), [statsmodels](https://www.statsmodels.org/stable/index.html)). In contast, for Bayesian statistics there has been [Bambi](https://bambinos.github.io/bambi/), which is excellent, but it does require more advanced knowledge and familiarity with R-brms syntax. Therefore, the goal of `bayes-toolbox` is to fill an important gap in the Python/Bayesian community, by providing an easy-to-use module for less experienced users that makes it as simple to run Bayesian stats as it is to run frequentist stats. As all of the models (tests) are executable with one-liners, they are ideal for use in an open, replicable (and Bayesian) workflow (watch this [PyMCon talk](https://www.youtube.com/watch?v=ElfToZ9EBpM) to learn more). **Example use cases and tests for nearly every model are provided in the [`examples`](https://github.com/hyosubkim/bayes-toolbox/tree/main/examples) directory, so you can see what a sensible Bayesian data analysis pipeline looks like.** (Many of the example notebooks are adaptations of [Jordi Warmenhoven's Python/PyMC3 port](https://github.com/JWarmenhoven/DBDA-python) of John Kruschke's excellent textbook ["Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan (DBDA)"](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0).

The benefits of using `bayes-toolbox` in terms of time and convenience will be most noticeable when utilizing hierarchical (multi-level) models, including the ANOVA-like ones. This is because `bayes-toolbox` takes care of the more finicky steps involved in Bayesian statistical modeling with embedded functions for things like standardizing/unstandardizing variables for more efficient MCMC sampling, parsing categorical variables for easier indexing, and implementing sum-to-zero constraints in ANOVA-like models. These are the sorts of implementational details that can add time (and frustration) when creating an analysis pipeline and discourage otherwise interested scientists from using Bayesian statistics. `bayes-toolbox` now removes those obstacles. 

The package is actively being developed in a public [GitHub repository](https://github.com/hyosubkim/bayes-toolbox), and we always welcome new contributors! No contribution is too small. If you have any issues with the code, suggestions on how to improve it, or have requests for additional content, whether new features or tutorials, please [open an issue on Github](https://github.com/hyosubkim/bayes-toolbox/issues). 

### Education
`bayes-toolbox` will be very useful if you are going through [Doing Bayesian Data Analysis](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0) and want to learn how to implement the models in Python/PyMC. We also highly recommend going through some of the [PyMC tutorials](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/index.html) to supplement your understanding. The PyMC developers have also adapted many ideas from the Kruschke text. 

## Look before you leap
Please note that the models in `bayes-toolbox` all utilize fairly uninformative, diffuse priors, which are, for the most part, the exact same ones used in the Kruschke text. New users or those with “prior paralysis” will likely be relieved to know that these diffuse, uninformative priors will not exert undue influence over posterior estimates and will likely satisfy skeptical reviewers. However, keep in mind that even though `bayes-toolbox` offers a streamlined interface for performing common statistical tests, the assumptions John Kruschke and we, the developers, make in these models may not be the ones you want to make for your particular question. Therefore, it's a good idea to go through the notebooks in the [`examples`](https://github.com/hyosubkim/bayes-toolbox/tree/main/examples) to make sure the model is appropriate for your applications. Part of the beauty of Bayesian modeling is its flexibility, so if you want to change priors/hyperpriors/etc., feel free to use `bayes-toolbox` as model scaffolding for a new bespoke model fit to your purpose. And consider making it a [contribution](https://hyosubkim.github.io/bayes-toolbox/how-to-contribute/)! You may also want to explore using [Bambi](https://bambinos.github.io/bambi/). 

## Dependencies
Some of the main libraries used in this project:

- [ArviZ](https://arviz-devs.github.io/arviz/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [PyMC](https://www.pymc.io/welcome.html)
- [seaborn](https://seaborn.pydata.org/)
- [Xarray](https://docs.xarray.dev/en/stable/)

## Testing and Functionality
In addition to thorough formal testing of the functions that make up `bayes-toolbox`, the statistical models have all been validated against known results (i.e., "ground truth"). Specifically, in the `examples` directory, you will find that each model has been run on the same data used in [DBDA](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0). All the results have been compared to those in the textbook, and against the results produced in another Python port of the textbook (https://github.com/JWarmenhoven/DBDA-python). Only subtle numerical discrepancies due to the nature of [MCMC sampling](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo), as well as differences between [RStan](https://cran.r-project.org/web/packages/rstan/vignettes/rstan.html) and [PyMC](https://www.pymc.io/welcome.html), have been detected. 


---
**NOTE**      
Use the links in the navigation bar to the left, the search bar in the upper left, or the content pages below to get started!  

---
[Getting Started](https://hyosubkim.github.io/bayes-toolbox/getting-started/)  
[Tutorials](https://hyosubkim.github.io/bayes-toolbox/tutorials/)  
[Future Plans](https://hyosubkim.github.io/bayes-toolbox/future-plans/)     
[How to Contribute](https://hyosubkim.github.io/bayes-toolbox/how-to-contribute/)

## Bayesian models currently included (frequentist analogue in parentheses)
- See [API Reference](https://hyosubkim.github.io/bayes-toolbox/reference) for comprehensive list
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
If you use `bayes-toolbox` in your work, please cite our Journal of Open Source Software (JOSS) article: 

***APA format:***    
Kim, H. E. (2023). bayes-toolbox: A Python package for Bayesian statistics. Journal of Open Source Software, 8(90), 5526. https://doi.org/10.21105/joss.05526
    
***BibTeX format:***
```
@article{Kim_bayes-toolbox_A_Python_2023,
author = {Kim, Hyosub E.},
doi = {10.21105/joss.05526},
journal = {Journal of Open Source Software},
month = oct,
number = {90},
pages = {5526},
title = {{bayes-toolbox: A Python package for Bayesian statistics}},
url = {https://joss.theoj.org/papers/10.21105/joss.05526},
volume = {8},
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


