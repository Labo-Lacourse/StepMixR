### =======================
### Interface to stepmix.
###
### Éric Lacourse
### Roxane de la Sablonnière
### Charles-Édouard Giguère (Maintainer)
### Sacha Morin
### Robin Legault
### Zsusza Bakk
### =======================

### Function stepmix. Kept as an R object. The python package is
### only called when we fit some data to it.
stepmix <- function(n_components = 2, n_steps = 1,
                    measurement = "bernoulli", structural = "gaussian_unit",
                    assignment = "modal", correction = NULL,
                    abs_tol = 1e-10, rel_tol = 0.0, max_iter = 1000,
                    n_init = 1, init_params = "random",
                    random_state = NULL, verbose = 0,
                    progress_bar = 1, measurement_params = NULL,
                    structural_params = NULL){

    ## If integer parameters are not Null we
    ## need to convert them to integer because by
    ## default R converts them to float.
    if(!is.null(random_state))
        random_state = as.integer(random_state)

    ## List object containing all parameters.
    sm_object = list(n_components = as.integer(n_components),
                     n_steps = as.integer(n_steps),
                     measurement = measurement,
                     structural = structural,
                     assignment = assignment,
                     correction = correction,
                     abs_tol = abs_tol, rel_tol = rel_tol,
                     max_iter = as.integer(max_iter),
                     n_init = as.integer(n_init),
                     init_params = init_params,
                     random_state = random_state,
                     verbose = as.integer(verbose),
                     progress_bar = as.integer(progress_bar),
                     measurement_params = measurement_params,
                     structural_params = structural_params)
    ## declare stepmixr object.
    class(sm_object) <- "stepmixr"
    sm_object
}

### Function to make latent class that uses mixed
### classes.
mixed_descriptor <- function(data, continuous = NULL,
                             binary = NULL,
                             categorical = NULL,
                             covariate = NULL){
    ## We put the drop = FALSE to protect R from converting it
    ## into a vector.
    data_mixed = data[,c(continuous, binary, categorical, covariate),
                      drop = FALSE]
    desc_mixed <- list()
    if(!is.null(continuous)){
        desc_mixed[["continuous"]] <- list(
            model = "continuous",
            n_columns = as.integer(length(continuous)))
    }
    if(!is.null(binary)){
        desc_mixed[["binary"]] <- list(
            model = "binary",
            n_columns = as.integer(length(binary)))
    }
    if(!is.null(categorical)){
        desc_mixed[["categorical"]] <- list(
            model = "categorical",
            n_columns = as.integer(length(categorical)))
    }
    if(!is.null(covariate)){
        desc_mixed[["covariate"]] <- list(
            model = "covariate",
            n_columns = as.integer(length(covariate)))
    }
    list(data = data_mixed, descriptor = desc_mixed)
}

### Function to print a stepmix object.
print.stepmixr <- function(x, ..., options = 1){
    if(options == 0){
        cat("StepMix()\n")
    }
    else if(options == 1){
        cat("Stepmix object: \n")
        if(x$n_steps == 1){
            cat(sprintf(" %d components, 1 step",x$n_components))
        }
        else{
            cat(sprintf(" %d components, %d steps",
                        x$n_components, x$n_steps))
        }
    }
    else {
        cat("StepMix()\n")
        print(unlist(x))
    }
}



### Function to fit a stepmix model. The object returned is
### a pointer to a python object. It cannot be saved using
### saveRDS or save command. To save a StepMix fitted object
### use the savefit and loadfit object.
fit <- function(smx, X = NULL, Y = NULL, ...){
    ## if both x and y are null, we return smx
    if(is.null(X) & is.null(Y)){
        stop("Both X and Y aren't specified")
    }
    if(is.null(X)){
        stop("X must be specified")
    }
    ## On fit X seulement.
    if(is.null(Y)){
        py_config <- try(reticulate::py_discover_config(required_module = "stepmix"))
        ## load
        sm <- try(reticulate::import("stepmix"), silent = TRUE)
        if(inherits(sm, "try-error"))
            stop(paste("Unable to find stepmix library in your python repos\n",
                       "Install it using pip install stepmix.",collapse = ""))
        model <- do.call(sm$StepMix, smx)
        fit <- model$fit(as.data.frame(X), ...)
        attr(fit, "X") <- X
        attr(fit, "Y") <- NULL
        return(fit)
    }
    else{
        py_config <- try(reticulate::py_discover_config(required_module = "stepmix"))
        ## load
        sm <- try(reticulate::import("stepmix"), silent = TRUE)
        if(inherits(stepmix,"try-error"))
            stop(paste("Unable to find stepmix library in your python repos\n",
                       "Install it using pip install stepmix.",collapse = ""))
        model <- do.call(sm$StepMix, smx)
        fit <- model$fit(as.data.frame(X), Y, ...)
        attr(fit, "X") <- X
        attr(fit, "Y") <- Y
        return(fit)
    }
}

### Predict the membership using fit. The function
### overloads the predict function for stepmix object.
predict.stepmix.stepmix.StepMix <- function(object, X = NULL, Y = NULL, ...){

    ## if both x and y are null, we return smx
    if(is.null(X) & is.null(Y)){
        stop("Both X and Y aren't specified")
    }
    if(is.null(X)){
        stop("X must be specified")
    }

    ## On fit X seulement.
    if(is.null(Y)){
        pr = object$predict(X)
    }
    else{
        pr = object$predict(X, Y)
    }
    return(pr)
}


### Predict the membership using fit. The function
### overloads the predict function for stepmix object.
predict_proba <- function(object, ...)
    UseMethod("predict_proba")

predict_proba.stepmix.stepmix.StepMix <- function(object, X = NULL, Y = NULL, ...){
    ## if both x and y are null, we return smx
    if(is.null(X) & is.null(Y)){
        stop("Both X and Y aren't specified")
    }

    if(is.null(X)){
        stop("X must be specified")
    }
    ## On fit X seulement.
    if(is.null(Y)){
        pr = object$predict_proba(X)
    }
    else{
        pr = object$predict_proba(X, Y)
    }
    return(pr)
}


### Print methods that replicate the ouput used when using verbose methods.
print.stepmix.stepmix.StepMix <- function(x, x_names = NULL, y_names = NULL, ...){
    sm <- try(reticulate::import("stepmix"), silent = TRUE)
    if(inherits(sm, "try-error"))
        stop(paste("Unable to find stepmix library in your python repos\n",
                   "Install it using pip install stepmix.",collapse = ""))
    if(is.null(attr(x, "Y"))){
        if(is.null(x_names)){
            x_names = names(attr(x, "X"))
        }
        sm$utils$print_report(x,attr(x, "X"), x_names = x_names, ...)
    }
    else{
        if(is.null(x_names)){
            x_names = names(attr(x, "X"))
        }
        if(is.null(y_names)){
            y_names = names(attr(x, "Y"))
        }
        sm$utils$print_report(x, attr(x, "X"), attr(x, "Y"), x_names = x_names,
                              y_names = y_names, ...)
    }
}

bootstrap <- function(x, ...)
    UseMethod("bootstrap")

bootstrap.stepmix.stepmix.StepMix <- function(x, X = NULL, y = NULL,
                                              n_repetitions = 10, ...){
    if(is.null(X)){
        stop("X value must be specified")
    }

    if(is.null(y)){
        result <- x$bootstrap(X, n_repetitions = as.integer(n_repetitions), ...)
    } else{
        result <- x$bootstrap(X, y, n_repetitions = as.integer(n_repetitions), ...)
    }
    list(samples = cbind(reticulate::py_to_r(attr(result[[1]], "pandas.index")$to_frame()),
                         result[[1]]),
         rep_stats = result[[2]])
}

bootstrap_stats <- function(x, ...)
    UseMethod("bootstrap_stats")

bootstrap_stats.stepmix.stepmix.StepMix <- function(x, X = NULL, y = NULL,
                                                    n_repetitions = 10, ...){
    if(is.null(X)){
        stop("X value must be specified")
    }
    if(is.null(y)){
        result <- x$bootstrap_stats(X, n_repetitions = as.integer(n_repetitions))
    } else{
        result <- x$bootstrap_stats(X, y, n_repetitions = as.integer(n_repetitions))
    }
    if(is.null(result['cw_mean'])){
        resl <- list(samples = cbind(reticulate::py_to_r(attr(result[[1]],
                                                              "pandas.index")$to_frame()),
                                     result[[1]]),
                     rep_stats = result[[2]],
                     mm_mean = result['mm_mean'],
                     mm_std = result['mm_std'],
                     sm_mean = result['sm_mean'],
                     sm_std = result['sm_std'])
    }
    else{
        resl <- list(samples = cbind(reticulate::py_to_r(attr(result[[1]],
                                                              "pandas.index")$to_frame()),
                                     result[[1]]),
                     rep_stats = result[[2]],
                     mm_mean = result['mm_mean'],
                     mm_std = result['mm_std'],
                     sm_mean = result['sm_mean'],
                     sm_std = result['sm_std'],
                     cw_mean = result['cw_mean'],
                     cw_std = result['cw_std'])
    }
    resl
}

### Find a reference configuration of the coefficients.
### Set a reference class with null coefficients for identifiability
identify_coef <- function(coef){
    second_coef = order(coef[,2])[2]
    coef - matrix(coef[second_coef,], nrow = dim(coef)[1], ncol = dim(coef)[2], byrow = TRUE)
}


### Save a StepMix fit using pickle via reticulate.
savefit <- function(fitx, f){
    f1 = file(f, "wb")
    reticulate::py_save_object(fitx, f)
    close(f1)
}

### Load a StepMix fit using pickle via reticulate.
loadfit <- function(f){
    reticulate::py_load_object(f)
}

### Series of function added for securities and to pass
### CRAN check.

### Check version of stepmix.
check_pystepmix_version <- function() {
    pyversion <- strsplit(pystepmix()$`__version__`, '\\.')[[1]]
    rversion <- strsplit(as.character(packageVersion("stepmixr")), '\\.')[[1]]
    major_version <- as.integer(rversion[1])
    minor_version <- as.integer(rversion[2])
    if (as.integer(pyversion[1]) < major_version) {
        warning(paste0("Python stepmix version ",
                       pystepmix()$`__version__`,
                       " is out of date (recommended: ",
                       major_version, ".", minor_version,
                       "). Please update with pip ",
                       "(e.g. ", reticulate::py_config()$python,
                       " -m pip install --upgrade stepmix) or stepmixR::install.stepmix()."))
        return(FALSE)
    } else if (as.integer(pyversion[2]) < minor_version) {
        warning(paste0("Python stepmix version ",
                       pystepmix()$`__version__`,
                       " is out of date (recommended: ",
                       major_version, ".",
                       minor_version,
                       "). Consider updating with pip ",
                       "(e.g. ", reticulate::py_config()$python,
                       " -m pip install --upgrade stepmix) or stepmixR::install.stepmix()."))
        return(FALSE)
    }
    return(TRUE)
}

### check if stepmix can be loaded
failed_pystepmix_import <- function(e) {
    message("Error loading Python module stepmix")
    message(e)
    result <- as.character(e)
    if (length(grep("ModuleNotFoundError: No module named 'stepmix'", result)) > 0 ||
        length(grep("ImportError: No module named stepmix", result)) > 0) {
        ## not installed
        if (utils::menu(c("Yes", "No"),
                        title="Install stepmix Python package with reticulate?") == 1) {
            install.stepmix()
        }
    } else if (length(grep("r\\-reticulate", reticulate::py_config()$python)) > 0) {
        ## installed, but envs sometimes give weird results
        message("Consider removing the 'r-reticulate' environment by running:")
        if (length(grep("virtualenvs", reticulate::py_config()$python)) > 0) {
            message("reticulate::virtualenv_remove('r-reticulate')")
        } else {
            message("reticulate::conda_remove('r-reticulate')")
        }
    }
}

### load stepmix library.
load_pystepmix <- function(){
    py_config <- try(reticulate::py_discover_config(required_module = "stepmix"))
    delay_load = list(on_load=check_pystepmix_version, on_error=failed_pystepmix_import)
    ## load
    pystepmix <- try(reticulate::import("stepmix", delay_load = delay_load))
    pystepmix
}

### install stepmix package in python.
install.stepmix <- function(envname = "r-reticulate", method = "auto",
                            conda = "auto", pip=TRUE, ...) {
    tryCatch({
        message("Attempting to install stepmix Python package with reticulate")
        reticulate::py_install("stepmix",
                               envname = envname, method = method,
                               conda = conda, pip=pip, ...
                               )
        message("Install complete. Please restart R and try again.")
    },
    error = function(e) {
        stop(paste0(
            "Cannot locate stepmix Python package, please install through pip ",
            "(e.g. ", reticulate::py_config()$python,
            " -m pip install --user stepmix) and then restart R."
        ))
    })
}

pystepmix <- NULL
.onLoad <- function(libname, pkgname) {
    pystepmix <<- load_pystepmix
}

