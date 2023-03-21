library(stepmixr)

# Simulate Data
# Z includes both the covariate (first column) and the response (second column)
datasim = data_bakk_complete(2000, sep_level=.8, nan_ratio=.25, random_state=42)

# Define the structural model
structural_descriptor = list(
  # Covariate
  covariate = list( 
    model= "covariate",
    n_columns= as.integer(1),
    method= "newton-raphson",
    max_iter = as.integer(1)
  ),
  # Response
  response = list(
    model= "gaussian_unit_nan", # Allow missing values
    n_columns = as.integer(1)
  )
)

# Fit StepMix Estimator
model = stepmix(n_components=3, measurement="binary_nan", structural=structural_descriptor, 
                n_steps=1, random_state=42)
fit1 = fit(model, datasim[[1]], datasim[[2]])

# Retrieve mean parameters
mus = fit1$get_parameters()[['structural']][['response']][['means']]
