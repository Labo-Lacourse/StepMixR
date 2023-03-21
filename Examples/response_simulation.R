library(stepmixr)

# Simulate data
datasim <- data_bakk_response(n_samples=2000, sep_level=.9, random_state=42)

# Fit StepMix Estimator
model = stepmix(n_components=3, measurement='binary', structural='gaussian_unit', 
                n_steps=1, random_state=42)
fit1 <- fit(model, datasim[[1]], datasim[[2]])

# Retrieve mean parameters
mus = fit1$get_parameters()[["structural"]][["means"]]
