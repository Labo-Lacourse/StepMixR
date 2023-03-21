library(stepmixr)

# Simulate data
datasim <- data_bakk_covariate(n_samples=2000, sep_level=.9, random_state=42)

# Specify optimization parameters for the covariate SM if needed
covariate_params <- list(method    = 'newton-raphson',
                         max_iter  = as.integer(1),
                         intercept = TRUE)

# Fit StepMix Estimator
model <- stepmix(n_components = 3, measurement = 'binary', 
                 structural = 'covariate', n_steps = 1, random_state = 42, 
                 structural_params = covariate_params)

fit1 = fit(model, datasim[[1]], datasim[[2]])

# Retrieve coefficients
betas = fit1$get_parameters()[['structural']][['beta']]

coef = betas


identify_coef(betas)
