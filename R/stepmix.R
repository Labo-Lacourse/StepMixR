### =======================
### Interface to stepmix.
### Charles-Édouard Giguère
### =======================

### Function stepmix. Kept as an R object. The python package is
### only called when we fit some data to it.
stepmix <- function(n_components = 2, n_steps = 1,
                    measurement = "bernoulli", structural = "bernoulli",
                    assignment = "modal", correction = NULL,
                    abs_tol = 1e-3, rel_tol = 1e-10, max_iter = 100,
                    n_init = 1, init_params = "random",
                    random_state = NULL, verbose = 0,
                    verbose_interval = 10, measurement_params = NULL,
                    structural_params = NULL){

  ## If integer parameters are not Null we
  ## need to convert them to integer because by
  ## default R converts them to float.
  if(!is.null(random_state))
    random_state = as.integer(random_state)
  if(!is.null(verbose_interval))
    verbose_interval = as.integer(verbose_interval)

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
                   verbose_interval = verbose_interval,
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
                             categorical = NULL){
  ## We put the drop = FALSE to protect R from converting it
  ## into a vector.
  data_mixed = data[,c(continuous, binary, categorical),
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
fit <- function(smx, X = NULL, Y = NULL){
  ## if both x and y are null, we return smx
  if(is.null(X) & is.null(Y)){
    stop("Both X and Y aren't specified")
  }

  if(is.null(X)){
    stop("X must be specified")
  }
  # On fit X seulement.
  if(is.null(Y)){
    py_config <- try(reticulate::py_discover_config(required_module = "stepmix"))
    # load
    sm <- try(reticulate::import("stepmix"), silent = TRUE)
    if(inherits(sm, "try-error"))
          stop(paste("Unable to find stepmix library in your python repos\n",
                     "Install it using pip install stepmix.",collapse = ""))
    model <- do.call(sm$StepMix, smx)
    fit <- model$fit(as.data.frame(X))
    return(fit)
  }
  else{
    py_config <- try(reticulate::py_discover_config(required_module = "stepmix"))
    # load
    sm <- try(reticulate::import("stepmix"), silent = TRUE)
    if(inherits(stepmix,"try-error"))
      stop(paste("Unable to find stepmix library in your python repos\n",
                 "Install it using pip install stepmix.",collapse = ""))
           model <- do.call(sm$StepMix, smx)
           fit <- model$fit(as.data.frame(X), as.data.frame(Y))
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
  # On fit X seulement.
  if(is.null(Y)){
    pr = object$predict(X)
  }
  else{
    pr = object$predict(X, Y)
  }
  return(pr)
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
    warning(paste0("Python stepmix version ", pystepmix()$`__version__`, " is out of date (recommended: ",
                   major_version, ".", minor_version, "). Please update with pip ",
                   "(e.g. ", reticulate::py_config()$python, " -m pip install --upgrade stepmix) or stepmixR::install.stepmix()."))
    return(FALSE)
  } else if (as.integer(pyversion[2]) < minor_version) {
    warning(paste0("Python stepmix version ", pystepmix()$`__version__`, " is out of date (recommended: ",
                   major_version, ".", minor_version, "). Consider updating with pip ",
                   "(e.g. ", reticulate::py_config()$python, " -m pip install --upgrade stepmix) or stepmixR::install.stepmix()."))
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
    # not installed
    if (utils::menu(c("Yes", "No"), title="Install stepmix Python package with reticulate?") == 1) {
      install.stepmix()
    }
  } else if (length(grep("r\\-reticulate", reticulate::py_config()$python)) > 0) {
    # installed, but envs sometimes give weird results
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
  # load
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
      "(e.g. ", reticulate::py_config()$python, " -m pip install --user stepmix) and then restart R."
    ))
  }
  )
}

pystepmix <- NULL

.onLoad <- function(libname, pkgname) {
  pystepmix <<- load_pystepmix
}

