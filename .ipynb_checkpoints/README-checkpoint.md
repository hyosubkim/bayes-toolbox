[![DOI](https://sandbox.zenodo.org/badge/553182204.svg)](https://sandbox.zenodo.org/badge/latestdoi/553182204)

# Bayesian Statistics Toolbox (BST) 

## What is this?
BST is a beta-version library of functions for running sophisticated Bayesian analyses in a simple, straight forward manner, and all in Python. 

BST provides you with the tools for utilizing and exploring Bayesian statistics in your own research projects right away. In addition, I have included example use cases for almost every model provided (in the `examples` directory), so you can see for yourself what a sensible Bayesian data analysis pipeline looks like. The example notebooks are primarily adaptations of [Jordi Warmenhoven's Python/PyMC3 port](https://github.com/JWarmenhoven/DBDA-python) of John Kruschke's excellent textbook ["Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan"](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0). In fact, BST is in large part updating Jordi Warmenhoven's original PyMC3 versions of the Kruschke models to [PyMC 4.0](https://www.pymc.io/welcome.html) and wrapping them into tidy functions so that they are easily re-usable. 

I've also tried to help take care of some of the more finicky steps involved in Bayesian statistical modeling by writing, adapting, and incorporating functions for things like standardizing/unstandardizing variables for more efficient MCMC sampling, parsing categorical variables for easier indexing, and implementing sum-to-zero constraints in ANOVA-like models. These are the sorts of implementational details that can add time when creating an analysis pipeline and make it less likely to use Bayesian statistics. I hope that BST removes those obstacles as well.  
 
## Why is this useful?
By wrapping model definitions of Bayesian generalized linear models into convenient functions, BST aims to make it easier to run Bayesian analyses that are analogous to some of the most commonly used frequentist tests in the behavioral and neural sciences (think t-tests, ANOVAs, regression). I originally started working on BST because I wanted to be able to utilize Bayesian statistics in my own research without having to code each model up from scratch, as that itself can be a barrier (especially when there is the temptation to fall back into frequentist habits using one-liners from `statsmodels`). Soon, I realized that this project may actually lower the bar to entry into the wonderful world of Bayesian statistics for others as well, and so here we are. 

BST is also useful if you're going through the DBDA textbook and want to see how to implement the models in PyMC 4.0. However, this project is still a work in progress and does not cover all of the models in DBDA. For a much more complete Python-ic implementation (in PyMC3), see Jordi Warmenhoven's [repo](https://github.com/JWarmenhoven/DBDA-python).

Please note that the models all utilize fairly uninformative, diffuse priors, which are, for the most part, the exact same ones used in the Kruschke text. An easy way to modify the priors, or any part of the model, for that matter, is to print the function (add two question marks after the function name) and copy it over to your editor. 

## Example syntax
Now, if you want to run a fairly sophisticated multi-level (hierarchical) linear regression model in which you are modeling individual as well as group-level slope and intercept parameters, it looks like

```python
# Call your BST function and return the PyMC model and InferenceData objects
model, idata = bst.hierarchical_regression(df["x"], df["y"], df["subj"], acceptance_rate=0.95)
```
Before, this would have taken *many* more lines of code. 


## Dependencies
Some of the main libraries used in this project:

- aesara
- arviz
- numpy
- pandas
- pymc
- seaborn

If you're running Mac OSX, and want to ensure you can run everything right out of the box, you may want to install a new virtual environment with all of the necessary dependencies as well. To do that, make sure you're in the root directory and type the following conda command in the Terminal:
```
$ conda env create --name bayes_toolbox --file environment.yml
```
You can name your environment whatever you like, it doesn't have to be "bayes_toolbox". 

If you're on a different OS and want to replicate this environment, read the "Export your environment" section of this [page](https://goodresearch.dev/setup.html). 

To make sure you can access the correct kernel from a Jupyter notebook, read this [link](https://janakiev.com/blog/jupyter-virtual-envs/).

## How do I get started? 
The `BEST`  notebook (short for "Bayesian Estimation Supersedes the t-Test", a famous 2013 [article](https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf) by John Kruschke) in `examples` is a good place to see how BST can be used to make Bayesian analyses more convenient. I've adapted the [notebook](https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html) of the same name from the PyMC developers to show how the model building and MCMC sampling are all embedded in a single function now. You can see similar workflows for other model types in the other example notebooks.

To access all of the functions, you can simply copy `bayesian_stats.py` in the `src` folder into your own analysis script. If you want to use it as a package, you can clone this repo and then pip install the package from the Terminal. Make sure you are in the correct directory and then at the prompt type in:
```
$ pip install -e .
```
Once installed locally, you can access it from any directory. When you import your packages, add the following line:

```python
import src.bayesian_stats as bst
```

## Where can I get more help?
If you're just starting off with Bayesian statistics, the two best introductory textbooks I know of are ["Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan"](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0) by John Kruschke and ["Statistical Rethinking"](https://xcelab.net/rm/statistical-rethinking/) by Richard McElreath. If you already have some background in Bayesian inference but are new to Python and/or PyMC, the PyMC developers have some great example notebooks [here](https://www.pymc.io/projects/examples/en/latest/gallery.html).

## Notice any bugs or have suggestions?
I welcome your suggestions for improvement. Feel free to open a new issue through GitHub. However, please keep in mind that I am a full-time academic [researcher](https://osf.io/y75ud/wiki/home/) and not a software developer, so I may not have the time or the know-how to implement even moderately complicated suggestions. 

## Models that are currently included and validated (frequentist analogue in parentheses)
- Comparison of two groups (independent samples t-test)
- Comparsion of single or paired samples (paired t-test)
- Simple linear regression
- Multiple regression
- Multi-level (hierarchical) linear regression
- Hierarchical (multi-level) model of metric outcome with single categorical predictor (one-way ANOVA)
- Hierarchical (multi-level) model of metric outcome with single categorical and single metric predictors (ANCOVA)
- Hierarchical (multi-level) model of metric outcome with two categorical predictors (two-way ANOVA)
- Hierarchical (multi-level) model of metric outcome with multiple categorical predictors and repeated measures (mixed-model ANOVA)

## Are there any plans to further develop BST?
Yes! I plan to add the following in the near future:

- Bayesian logistic regression 
- Bayes Factors
- Bayesian meta-analysis 
- Incorporating prior and posterior predictive checks into the examples
- Incorporating maximum entropy priors 

## Other related Python projects 
For a more weapons-grade Bayesian statistical modeling interface, check out:
- [Bambi](https://github.com/bambinos/bambi): BAyesian Model-Building Interface (BAMBI) in Python.

## Acknowledgments
Thanks to the following people for generously sharing their knowledge:
- [John Kruschke](https://jkkweb.sitehost.iu.edu/)
- [Richard McElreath](https://xcelab.net/rm/)
- [Jordi Warmenhoven](https://github.com/JWarmenhoven)
- [PyMC developers](https://github.com/pymc-devs/pymc)

## License
https://coderefinery.github.io/github-without-command-line/doi/#step-2-activate-the-repository-on-zenodo-sandbox

## Citations
https://citation-file-format.github.io/


