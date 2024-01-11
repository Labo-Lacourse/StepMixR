library(stepmixr)

if (reticulate::py_module_available("stepmix")){
  sim1 <- data_gaussian_diag(n_samples=50, sep_level=0.5, n_mm = 6, nan_ratio = 0.1)
}

if (reticulate::py_module_available("stepmix")) {
  require(stepmixr)
  model1 <- stepmix(n_components = 3, n_steps = 2, measurement = "continuous", verbose = 0,
                    progress_bar = 0)
  X <- iris[c(1:10, 51:60, 101:110), 1:4]
  fit1 <- fit(model1, X)
  fit1_bs <- bootstrap(fit1, X, n_repetitions = 5, progress_bar = FALSE)
}


if (reticulate::py_module_available("stepmix")) {
  model1 <- stepmix(n_components = 3, n_steps = 2, measurement = "continuous")
  X <- iris[c(1:10, 51:60, 101:110), 1:4]
  fit1 <- fit(model1, X)
}

md <- mixed_descriptor(iris, continuous = 1:4, categorical = 5)

if (reticulate::py_module_available("stepmix")) {
  model1 <- stepmix(n_components = 3, n_steps = 2, measurement = "continuous", progress_bar = 0)
  X <- iris[c(1:10, 51:60, 101:110), 1:4]
  fit1 <- fit(model1, X)
  pr1 <- predict(fit1, X)
}

if (reticulate::py_module_available("stepmix")) {
  model1 <- stepmix(n_components = 2, n_steps = 3)
  X <- data.frame(x1 = c(0,1,1,1,1,0,0,0,0,0,1,1,0),
                  x2 = c(0,1,1,0,0,1,1,0,0,0,1,0,1))
  fit1 <- fit(model1, X)
  savefit(fit1, "fit1.pickle")
  
  ### clean the directory.
  file.remove("fit1.pickle")
}
