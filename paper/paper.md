---
title: 'bayes-toolbox: A Python package for Bayesian statistics'
tags:
  - Python
  - Bayesian statistics
  - psychology
  - neuroscience
authors:
  - name: Hyosub E. Kim
    orcid: 0000-0003-0109-593X
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: School of Kinesiology, The University of British Columbia, Canada 
   index: 1
 - name: Department of Physical Therapy, University of Delaware, United States
   index: 2
date: 22 April 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`bayes-toolbox` is a Python package intended to facilitate the increased use and adoption of Bayesian statistics in scientific research. As Python is one of the fastest growing and most widely used programming languages, `bayes-toolbox` fills the need for a Python library that makes it as easy to perform Bayesian statistical tests as it currently is to perform their "frequentist" counterparts. The intended users of `bayes-toolbox` are students and researchers who are looking for a low-friction way to learn Bayesian statistics and incorporate it into their research.

# Statement of need

Given the current "replication crisis" in science, and increasing recognition of how the improper usage of statistics has contributed to this state, there is a growing need for developing open, replicable data analysis pipelines that implement best practices [shrout2018psychology]. Among these best practices is the full quantification of uncertainty around parameter estimates and probability statements related to specific hypotheses (i.e., not only the null) [@wasserstein2016asa], features readily provided by Bayesian statistics [@kruschke2018bayesian]. `bayes-toolbox`is a Python package that makes performing such Bayesian analyses simple and straight forward. By leveraging PyMC, a probabilistic programming library written in Python [@patil2010pymc], and providing easy-to-use functions, `bayes-toolbox` removes many of the technical barriers previously associated with Bayesian analyses, especially for users wishing to work within the Python data science ecosystem, as opposed to in R. And as the `bayes-toolbox` functions provide Bayesian analogues of many of the most common "frequestist" tests used by scientists, including t-tests, ANOVAs, and regression models, as well as hierarchical (multi-level) models and meta-analyses, it provides a necessary bridge for researchers who are familiar with frequentist statistics but want to explore the Bayesian framework. 

`bayes-toolbox` was designed for and targeted to researchers primarily in the behavioral and neural sciences. However, as many of the models and Jupyter notebook tutorials included in the public `bayes-toolbox` repository are adapted from the well-known textbook "Doing Bayesian Data Analysis" [@kruschke2014doing], `bayes-toolbox` can also serve as an important pedagogical tool for students and researchers alike who wish to utilize Bayesian statistics in Python. 

# Acknowledgements

We thank the PyMC developers, John Kruschke, and Jordi Warmenhoven for generously sharing their work and knowledge. 

# References