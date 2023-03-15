## How do I get started?
*(Recommended)*   
If you're running Mac OSX, and want to ensure you can run everything right out of the box, after cloning [this repo](https://github.com/hyosubkim/bayesian-statistics-toolbox) you can create a virtual environment with all of the necessary dependencies. To do that, make sure you're in the root directory of this repository (i.e., `bayesian-statistics-toolbox`) and type the following conda command in the Terminal (I strongly recommend [Anaconda](https://www.anaconda.com/) to install Python and the conda utility on your computer):
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
The `BEST`  notebook (short for "Bayesian Estimation Supersedes the t-Test", a famous 2013 [article](https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf) by John Kruschke) in the `examples` [directory](https://github.com/hyosubkim/bayesian-statistics-toolbox/tree/main/examples) is a good place to see how BST can be used to make Bayesian analyses more convenient. I've adapted the [notebook](https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html) of the same name from the PyMC developers to show how the model building and MCMC sampling are all embedded in a single function now. You can see similar workflows for other model types in all of the other example notebooks, which track several of the chapters from "Doing Bayesian Data Analysis" and is modeled off of Jordi Warmenhoven's [repo](https://github.com/JWarmenhoven/DBDA-python).

## Example syntax
Now, if you want to run a fairly sophisticated multi-level (hierarchical) linear regression model in which you are modeling individual as well as group-level slope and intercept parameters, it looks like

```python
# Call your BST function and return the PyMC model and InferenceData objects
model, idata = bst.hierarchical_regression(df["x"], df["y"], df["subj"], acceptance_rate=0.95)
```
Before, this would have taken *many* more lines of code. 