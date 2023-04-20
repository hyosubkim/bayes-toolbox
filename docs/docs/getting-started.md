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
git clone git://github.com/hyosubkim/bayes-toolbox.git
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
The `BEST`  notebook (short for "Bayesian Estimation Supersedes the t-Test", a famous 2013 [article](https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf) by John Kruschke) in the `examples` [directory](https://github.com/hyosubkim/bayesian-statistics-toolbox/tree/main/examples) is a good place to see how `bayes-toolbox` can be used to make Bayesian analyses simple to implement. I've adapted the [notebook](https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html) of the same name from the PyMC developers to show how the model building and MCMC sampling are all embedded in a single function now. You can see similar workflows for other model types in all of the other example notebooks, which track several of the chapters from "Doing Bayesian Data Analysis" and is modeled off of Jordi Warmenhoven's [repo](https://github.com/JWarmenhoven/DBDA-python).

## Example syntax
Now, if you want to run a fairly sophisticated multi-level (hierarchical) linear regression model in which you are modeling individual as well as group-level slope and intercept parameters, it looks like

```python
# Import the bayes-toolbox package 
import bayes_toolbox.glm as bg

# Call your bayes-toolbox function and return the PyMC model and InferenceData objects
model, idata = bg.hierarchical_regression(df["x"], df["y"], df["subj"], acceptance_rate=0.95)
```
Before, this would have taken *many* more lines of code. 