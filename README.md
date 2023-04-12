# StepMixR : R interface to Python package StepMix

<!-- badges: start -->
[![Build Status](https://api.travis-ci.com/stepmixr/stepmixr.svg?branch=master)](https://app.travis-ci.com/github/stepmixr/stepmixr)
[![Codecov test coverage](https://codecov.io/gh/stepmixr/OpenMx/branch/master/graph/badge.svg)](https://app.codecov.io/gh/stepmixr/Ostepmixr?branch=master)
[![cran version](http://www.r-pkg.org/badges/version/stepmixr)](https://cran.r-project.org/package=stepmixr)
[![Monthly Downloads](https://cranlogs.r-pkg.org/badges/stepmixr)](https://cranlogs.r-pkg.org/badges/stepmixr)
[![Total Downloads](https://cranlogs.r-pkg.org/badges/grand-total/stepmixr)](https://cranlogs.r-pkg.org/badges/grand-total/stepmixr)
<!-- badges: end -->

For StepMix, please refer to <a href="https://github.com/Labo-Lacourse/stepmix">this repository.</a>

This is an R interface to Stepmix, a Python package following the scikit-learn API for model-based clustering and generalized mixture modeling (latent class/profile analysis) of continuous and categorical data. StepMix handles missing values through Full Information Maximum Likelihood (FIML) and provides multiple stepwise Expectation-Maximization (EM) estimation methods based on pseudolikelihood theory. Additional features include support for covariates and distal outcomes, various simulation utilities, and non-parametric bootstrapping, which allows inference in semi-supervised and unsupervised settings.

# Reference
If you find StepMix useful, please consider citing our [arXiv preprint](https://arxiv.org/abs/2304.03853):
```
@article{morin2023stepmix,
  title={StepMix: A Python Package for Pseudo-Likelihood Estimation of Generalized Mixture Models with External Variables},
  author={Morin, Sacha and Legault, Robin and Bakk, Zsuzsa and Gigu{\`e}re, Charles-{\'E}douard and de la Sablonni{\`e}re, Roxane and Lacourse, {\'E}ric},
  journal={arXiv preprint arXiv:2304.03853},
  year={2023}
}
```

# Install
You can install StepMixR from [CRAN](https://cran.r-project.org/web/packages/stepmixr/index.html) inside r using the function install.packages: 
```
install.packages("stepmixr")
```
To install directly from github you need to have the package `devtools` installed. Once it is installed, you can use the following syntax.

```
devtools::install_github("Labo-Lacourse/stepmixr")
```

# Tutorials

1. [A notebook available from google colab](https://colab.research.google.com/drive/1MzGHRO5kfs9OT3cRICJ1Ey94PHHnxFdO#scrollTo=b4T6zgxtbmY_) gives a detailed tutorials based on the iris dataset.  This notebook is a R adaptation of a similar Python notebook which can be found [here](https://colab.research.google.com/drive/1KAxcvxjL_vB2lAG9e47we7hrf_2fR1eK?usp=sharing). This tutorial covers : 
    1. Continuous LCA models;
    2. Binary LCA models;
    3. Categorical LCA models;
    4. Mixed LCA models (continuous and categorical data);
    5. Missing values.

# Quickstart

Here is a quick example from R documentation. 

```
  model1 <- stepmix(n_components = 3, n_steps = 2, measurement = "continuous")
  X <- iris[, 1:4]
  fit1 <- fit(model1, X)
  pr1 <- predict(fit1, X)
```
