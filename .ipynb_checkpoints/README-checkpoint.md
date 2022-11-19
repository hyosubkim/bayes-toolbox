# Bayesian Statistics Toolbox (BST) 

- What does this project do?
- Why is this project useful?
- How do I get started?
- Where can I get more help, if I need it?
- Goals
- Contributions

## What is this?
BST is a library of functions for running sophisticated Bayesian analyses in a simple, straight forward manner, and all in Python. 

BST provides you with the tools for utilizing and exploring Bayesian statistics in your own research projects right away. In addition, I have included example use cases for almost every model provided (in the `examples` directory), so you can see for yourself what a sensible Bayesian data analysis pipeline looks like. The example notebooks are adaptations of [Jordi Warmenhoven's Python/PyMC3 port](https://github.com/JWarmenhoven/DBDA-python) of John Kruschke's excellent textbook ["Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan"](https://sites.google.com/site/doingbayesiandataanalysis/home?authuser=0). Please note that these notebooks have been updated to make use of PyMC4 and that this does not (yet) include all of the examples from the original.

Tests were performed on Kruschke data. 

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


## Where can I get more help?


## Goals

## Contributions
John Kruschke
Richard McElreath
Jordi Warmhoven
PyMC developers


## License


## Citations


