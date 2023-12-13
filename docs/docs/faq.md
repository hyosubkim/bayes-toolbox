# Frequently Asked Questions

## General
---

### How come `bayes-toolbox` is not installing on my computer?

First, make sure you've followed all of the instructions on the [Getting Started](https://hyosubkim.github.io/bayes-toolbox/getting-started/) page, including the instructions on creating a virtual environment with the necessary dependencies. 

Note that as of December 2023, you cannot `pip install bayes_toolbox` with Python 3.12 or above. You can downgrade to Python 3.11 or, better yet, follow the [Getting Started](https://hyosubkim.github.io/bayes-toolbox/getting-started/) instructions and install the dependencies using the *environment.yml* file provided with `bayes-toolbox`.  The issue with pip and Python 3.12 is not specific to `bayes-toolbox`&mdash;you can read more about it [here](https://stackoverflow.com/questions/77364550/attributeerror-module-pkgutil-has-no-attribute-impimporter-did-you-mean).

If you continue to have problems with installation, please open a [GitHub Issue](https://github.com/hyosubkim/bayes-toolbox/issues).