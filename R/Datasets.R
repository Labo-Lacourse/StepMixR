### =================================================
### R interface to Datasets sub module of stepmix.
###
### Éric Lacourse
### Roxane de la Sablonnière
### Charles-Édouard Giguère (Maintainer)
### Sacha Morin
### Robin Legault
### Zsusza Bakk
### =================================================

### Randomly replace values in X and Y with NaNs with probability nan_ratio.
random_nan <- function(X, Y, nan_ratio, random_state=NULL){
    sm <- try(reticulate::import("stepmix"), silent = TRUE)
    if(!is.null(random_state))
        random_state = as.integer(random_state)
    if(inherits(sm, "try-error"))
        stop(paste("Unable to find stepmix library in your python repos\n",
                   "Install it using pip install stepmix.",collapse = ""))
    sm$datasets$random_nan(X,Y,nan_ratio, as.integer(random_state))
}

###  Binary measurement parameters in Bakk 2018.
###    Parameters
###    ----------
###    n_classes: int
###        Number of latent classes. Use 3 for the paper simulation.
###    n_mm: int
###        Number of features in the measurement model. Use 6 for the paper simulation.
###    sep_level : float
###        Separation level in the measurement data. Use .7, .8 or .9 for the paper simulation.
###    Returns
###    -------
###    pis : ndarray of shape (n_mm, n_classes)
###        Conditional bernoulli probabilities.

bakk_measurements <- function(n_classes, n_mm, sep_level){
    sm <- try(reticulate::import("stepmix"), silent = TRUE)
    if(inherits(sm, "try-error"))
        stop(paste("Unable to find stepmix library in your python repos\n",
                   "Install it using pip install stepmix",collapse = ""))
    sm$datasets$bakk_measurements(as.integer(n_classes), as.integer(n_mm), sep_level)
}


### Simulated data for the response simulations in Bakk 2018.
###     Parameters
###     ----------
###     n_samples : int
###         Number of samples.
###     sep_level : float
###         Separation level in the measurement data. Use .7, .8 or .9 for the paper simulation.
###     n_classes: int
###         Number of latent classes. Use 3 for the paper simulation.
###     n_mm: int
###         Number of features in the measurement model. Use 6 for the paper simulation.
###     random_state: int
###         Random state.
###     Returns
###     -------
###     X : ndarray of shape (n_samples, n_mm)
###         Binary measurement samples.
###     Y : ndarray of shape (n_samples, 1)
###         Response structural samples.
###     labels : ndarray of shape (n_samples,)
###         Ground truth class membership.

data_bakk_response <- function(n_samples, sep_level, n_classes = 3, n_mm = 6, random_state = NULL){
    sm <- try(reticulate::import("stepmix"), silent = TRUE)
    if(!is.null(random_state))
        random_state = as.integer(random_state)
    if(inherits(sm, "try-error"))
        stop(paste("Unable to find stepmix library in your python repos\n",
                   "Install it using pip install stepmix",collapse = ""))
    sm$datasets$data_bakk_response(as.integer(n_samples), sep_level, as.integer(n_classes),
                                   as.integer(n_mm), as.integer(random_state))
}


### Simulated data for the covariate simulations in Bakk 2018.
###     Parameters
###     ----------
###     n_samples : int
###         Number of samples.
###     sep_level : float
###         Separation level in the measurement data. Use .7, .8 or .9 for the paper simulation.
###     n_mm: int
###         Number of features in the measurement model. Use 6 for the paper simulation.
###     random_state: int
###         Random state.
###     Returns
###     -------
###     X : ndarray of shape (n_samples, n_mm)
###         Binary measurement samples.
###     Y : ndarray of shape (n_samples, 1)
###         Covariate structural samples.
###     labels : ndarray of shape (n_samples,)
###         Ground truth class membership.

data_bakk_covariate <- function(n_samples, sep_level, n_mm = 6, random_state = NULL){
    sm <- try(reticulate::import("stepmix"), silent = TRUE)
    if(!is.null(random_state))
        random_state = as.integer(random_state)
    if(inherits(sm, "try-error"))
        stop(paste("Unable to find stepmix library in your python repos\n",
                   "Install it using pip install stepmix",collapse = ""))
    sm$datasets$data_bakk_covariate(as.integer(n_samples), sep_level, as.integer(n_mm), random_state)
}

### Stitch together data_bakk_covariate and data_bakk_response to get a complete model.

data_bakk_complete <- function(n_samples, sep_level, n_mm=6, random_state=NULL, nan_ratio=0.0){
    sm <- try(reticulate::import("stepmix"), silent = TRUE)
    if(!is.null(random_state))
        random_state = as.integer(random_state)
    if(inherits(sm, "try-error"))
        stop(paste("Unable to find stepmix library in your python repos\n",
                   "Install it using pip install stepmix",collapse = ""))
    sm$datasets$data_bakk_complete(as.integer(n_samples), sep_level, as.integer(n_mm), random_state,
                                   nan_ratio)
}


### Bakk binary measurement model with more complex gaussian structural model.
###     Parameters
###     ----------
###     n_samples : int
###         Number of samples.
###     sep_level : float
###         Separation level in the measurement data. Use .7, .8 or .9 for the paper simulation.
###     n_mm: int
###         Number of features in the measurement model. Use 6 for the paper simulation.
###     random_state: int
###         Random state.
###     Returns
###     -------
###     X : ndarray of shape (n_samples, n_mm)
###         Binary Measurement samples.
###     Y : ndarray of shape (n_samples, 2)
###         Gaussian Structural samples.
###     labels : ndarray of shape (n_samples,)
###         Ground truth class membership.

data_generation_gaussian <- function(n_samples, sep_level, n_mm=6, random_state=NULL){
    sm <- try(reticulate::import("stepmix"), silent = TRUE)
    if(!is.null(random_state))
        random_state = as.integer(random_state)
    if(inherits(sm, "try-error"))
        stop(paste("Unable to find stepmix library in your python repos\n",
                   "Install it using pip install stepmix",collapse = ""))
    sm$datasets$data_generation_gaussian(as.integer(n_samples), sep_level,
                                         as.integer(n_mm), random_state)
}


### Bakk binary measurement model with 2D diagonal gaussian structural model.
###     Optionally, a random proportion of values can be replaced with missing values
###     to test FIML models.
###     Parameters
###     ----------
###     n_samples : int
###         Number of samples.
###     sep_level : float
###         Separation level in the measurement data. Use .7, .8 or .9 for the paper simulation.
###     n_mm: int
###         Number of features in the measurement model. Use 6 for the paper simulation.
###     random_state: int
###         Random state.
###     nan_ratio: float
###         Ratio of values to replace with missing values.
###     Returns
###     -------
###     X : ndarray of shape (n_samples, n_mm)
###         Binary ,easurement samples.
###     Y : ndarray of shape (n_samples, 2)
###         Gaussian structural samples.
###     labels : ndarray of shape (n_samples,)
###         Ground truth class membership.


data_gaussian_diag <- function(n_samples, sep_level, n_mm = 6, random_state = NULL, nan_ratio = 0.0){
    sm <- try(reticulate::import("stepmix"), silent = TRUE)
    if(!is.null(random_state))
        random_state = as.integer(random_state)
    if(inherits(sm, "try-error"))
        stop(paste("Unable to find stepmix library in your python repos\n",
                   "Install it using pip install stepmix",collapse = ""))
    sm$datasets$data_gaussian_diag(as.integer(n_samples), sep_level,
                                         as.integer(n_mm), random_state, nan_ratio)
}

