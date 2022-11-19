# Bayesian Statistics Toolbox (BST) 

- What does this project do?
- Why is this project useful?
- How do I get started?
- Where can I get more help, if I need it?
- Goals
- Contributions

## What is this?
BST is a beta-version library of functions for running sophisticated Bayesian analyses in a simple, straight forward manner, and all in Python.  

BST provides you with the tools for utilizing and exploring Bayesian statistics in your own research projects right away. In addition, I have included example use cases for almost every model provided (in the `examples` directory), so you can see for yourself what a sensible Bayesian data analysis pipeline looks like. The example notebooks are primarily adaptations of [Jordi Warmenhoven's Python/PyMC3 port](https://github.com/JWarmenhoven/DBDA-python) of John Kruschke's excellent textbook ["Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan"](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0). In fact, BST is in large part updating Jordi Warmenhoven's original PyMC3 versions of the Kruschke models to [PyMC 4.0](https://www.pymc.io/welcome.html) and wrapping them into tidy functions so that they are easily re-usable. 
 
## Why is this useful?
By wrapping model definitions of Bayesian generalized linear models into convenient functions, BST aims to make it easier to run Bayesian analyses that are analogous to some of the most commonly used frequentist tests in the behavioral and neural sciences (think t-tests, ANOVAs, regression). I originally started working on BST because I wanted to be able to utilize Bayesian statistics in my own research without having to code each model up from scratch, as that itself can be a barrier (especially when there is the temptation to fall back into frequentist habits using one-liners from `statsmodels`). Soon, I realized that this project may actually lower the bar to entry into the wonderful world of Bayesian statistics for others as well, and so here we are. 

## Dependencies
Some of the main libraries used in this project:

- aesara
- arviz
- numpy
- pandas
- pymc
- seaborn

If you're running Mac OSX, and want to ensure you can run everything right out of the box, you may want to install a new virtual environment with all of the necessary dependencies as well. To do that, make sure you're in the root directory and type the following conda command in the Terminal:

$ conda env create --name bayes_toolbox --file environment.yml

You can name your environment whatever you like, it doesn't have to be "bayes_toolbox". 

If you're on a different OS and want to replicate this environment, read the "Export your environment" section of this [page](https://goodresearch.dev/setup.html). 

To make sure you can access the correct kernel from a Jupyter notebook, read this [link](https://janakiev.com/blog/jupyter-virtual-envs/).

## How do I get started? 
The `BEST`  notebook (short for "Bayesian Estimation Supersedes the t-Test", a famous 2013 [article](https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf) by John Kruschke) in `examples` is a good place to see how BST can be used to make Bayesian analyses more convenient. I've adapted the [notebook](https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html) of the same name from the PyMC developers to show how the model building and MCMC sampling are all embedded in a single function now. You can see similar workflows for other model types in the other example notebooks.

To access all of the functions, you can simply copy `bayesian_stats.py` in the `src` folder into your own analysis script. If you want to use it as a package, you can clone this repo and then pip install the package from the Terminal. Make sure you are in the correct directory and then at the prompt type in:

$ pip install -e .

Once installed locally, you can access it from any directory. When you import your packages, add the following line:

`import src.bayesian_stats as bst`

## Where can I get more help?


## Goals

## Contributions
John Kruschke
Richard McElreath
Jordi Warmhoven
PyMC developers


## License


## Citations


