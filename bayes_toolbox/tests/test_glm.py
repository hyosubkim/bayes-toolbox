import arviz as az
import numpy as np
import pandas as pd
import pytest
import pymc as pm
from bayes_toolbox.glm import *

def test_mus():
    X = np.random.normal(loc=10, scale=20, size=(10,2))
    X_s, mu_X, _ = standardize(X)
    assert np.allclose(X_s.mean(axis=0), np.zeros_like(mu_X), atol=1e-4)
    
def test_sigmas():
    X = np.random.normal(loc=10, scale=20, size=(10,2))
    X_s, _, sigma_X = standardize(X)
    assert np.allclose(X_s.std(axis=0), np.ones_like(sigma_X), atol=1e-4)

def test_parse_categorical():
    # Create a sample categorical variable
    x = pd.Series(["A", "B", "A", "C", "B", "C"]).astype("category")

    # Call the parse_categorical function with the sample variable
    categorical_values, levels, n_levels = parse_categorical(x)

    # Perform assertions to check if the function output is as expected
    expected_categorical_values = np.array([0, 1, 0, 2, 1, 2])
    expected_levels = pd.Categorical(["A", "B", "C"]).categories
    expected_n_levels = 3

    assert np.all(categorical_values == expected_categorical_values)
    assert list(levels) == list(expected_levels)
    assert n_levels == expected_n_levels

def test_BEST():
    # Creat test data
    iq_drug = np.array([
    101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105, 103, 
    100, 95, 102, 106, 109, 102, 82, 102, 100, 102, 102, 101, 102, 102,
    103, 103, 97, 97, 103, 101, 97, 104, 96, 103, 124, 101, 101, 100,
    101, 101, 104, 100, 101
    ])

    iq_placebo = np.array([
    99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102, 100, 105,
    88, 101, 100, 104, 100, 100, 100, 101, 102, 103, 97, 101, 101, 100,
    101, 99, 101, 100, 100, 101, 100, 99, 101, 100, 102, 99, 100, 99
    ])

    df1 = pd.DataFrame({"iq": iq_drug, "group": "drug"})
    df2 = pd.DataFrame({"iq": iq_placebo, "group": "placebo"})
    indv = pd.concat([df1, df2]).reset_index()

    # Call the BEST test with test data
    model, idata = BEST(indv["iq"], indv["group"])

    # Perform assertions to check if the function output is as expected
    assert isinstance(model, pm.Model)
    assert isinstance(idata, az.data.inference_data.InferenceData)
    assert "group_mean" in model.named_vars
    assert "group_std" in model.named_vars
    assert "nu_minus_one" in model.named_vars
    assert "nu" in model.named_vars
    assert "difference of means" in model.named_vars
    assert "effect size" in model.named_vars
        
def test_BEST_paired():
    # Generate test data
    np.random.seed(0)
    y1 = np.random.normal(0, 1, 100)
    y2 = np.random.normal(0.1, 1, 100)

    # Call the BEST_paired function with test data
    model, idata = BEST_paired(y1, y2)

    # Perform assertions to check if the function output is as expected
    assert isinstance(model, pm.Model)
    assert isinstance(idata, az.data.inference_data.InferenceData)
    assert "mu" in model.named_vars
    assert "sigma" in model.named_vars
    assert "nu_minus_one" in model.named_vars
    assert "nu" in model.named_vars
    assert "likelihood" in model.named_vars
    assert "effect_size" in model.named_vars
    
def test_robust_linear_regression():
    # Generate test data
    np.random.seed(0)
    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 1, 100)

    # Call the robust_linear_regression function with test data
    model, idata = robust_linear_regression(x, y)

    # Perform assertions to check if the function output is as expected
    assert isinstance(model, pm.Model)
    assert isinstance(idata, az.data.inference_data.InferenceData)
    assert "zbeta0" in model.named_vars
    assert "zbeta1" in model.named_vars
    assert "sigma" in model.named_vars
    assert "nu_minus_one" in model.named_vars
    assert "nu" in model.named_vars
    assert "nu_log10" in model.named_vars
    assert "likelihood" in model.named_vars
    
def test_hierarchical_regression():
    # Generate test data
    np.random.seed(0)
    x = np.random.normal(0, 1, 100)
    y = 2 * x + np.random.normal(0, 1, 100)
    subj = pd.Series(np.random.choice(["A", "B"], 100))

    # Call the robust_linear_regression function with test data
    model, idata = hierarchical_regression(x, y, subj)

    # Perform assertions to check if the function output is as expected
    assert isinstance(model, pm.Model)
    assert isinstance(idata, az.data.inference_data.InferenceData)
    assert "zbeta0" in model.named_vars
    assert "zbeta1" in model.named_vars
    assert "zsigma" in model.named_vars
    assert "nu" in model.named_vars

def test_hierarchical_bayesian_anova():
    # Generate test data
    x = pd.Series(['A', 'B', 'A', 'C', 'B', 'C'])
    y = np.random.normal(0, 1, 6)

    # Call the hierarchical_bayesian_anova function with test data
    model, idata = hierarchical_bayesian_anova(x, y)

    # Perform assertions to check if the function output is as expected
    assert isinstance(model, pm.Model)
    assert isinstance(idata, az.data.inference_data.InferenceData)
    assert "sigma_a" in model.named_vars
    assert "a0" in model.named_vars
    assert "a" in model.named_vars
    assert "sigma_y" in model.named_vars
    assert "likelihood" in model.named_vars
    assert "m" in model.named_vars
    assert "b0" in model.named_vars
    assert "b" in model.named_vars



