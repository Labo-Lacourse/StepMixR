### Install latest version.
devtools::install_github("Labo-Lacourse/stepmixr")



random_nan(iris, iris, 0.2, 2020)

### Load library
### If you need to set some path to python 
### you should do so before loading stepmixr.
library(stepmixr)

### Reinstall latest version of stepmix on python.
system("pip install stepmix")

# head of iris
head(iris)

continuous_data = iris[,1:4]
head(continuous_data)

# Create a continuous latent class model
model <- stepmix(n_components=3, measurement="continuous", verbose=1,
                random_state=123)
# Fit to data
fit1 <- fit(model, continuous_data)

sm <- try(reticulate::import("stepmix"), silent = TRUE)
if(inherits(sm, "try-error"))
  stop(paste("Unable to find stepmix library in your python repos\n",
             "Install it using pip install stepmix.",collapse = ""))

print(fit1)
sm$utils$print_report(fit1,continuous_data)

data_bakk_covariate <- function(n_samples, sep_level, n_mm = 6, random_state = NULL){
  sm <- try(reticulate::import("stepmix"), silent = TRUE)
  if(!is.null(random_state))
    random_state = as.integer(random_state)
  if(inherits(sm, "try-error"))
    stop(paste("Unable to find stepmix library in your python repos\n",
               "Install it using pip install stepmix",collapse = ""))
  sm$datasets$data_bakk_covariate(as.integer(n_samples), sep_level, as.integer(n_mm), random_state)
}

data_bakk_covariate(60, 0.6, 5, 1203)


# Make a copy of data iris.
iris <- iris

# Save class membership predictions to df
iris[,'continuous_pred'] <- predict(fit, continuous_data)

# Only a f
xtabs(~Species + continuous_pred, iris)


### Install fossil package if it is not already installed.
if(!("fossil" %in% installed.packages()[,1]))
  install.packages("fossil")
fossil::rand.index(as.numeric(iris$Species), iris$continuous_pred)

# Create binarized features based on quantiles
binary_data = sapply(continuous_data, function(x) (0:1)[cut(x,breaks = 2)])
head(binary_data)

# Binary latent class model
model = stepmix(n_components=3, measurement="binary", verbose=1, random_state=123)

# Fit model
fit_model <- fit(model, binary_data)

# Class predictions
iris[,'binary_pred'] = predict(fit_model, X = binary_data)

xtabs(~Species + binary_pred, iris)

fossil::rand.index(as.numeric(iris$Species), iris$binary_pred)

# Create categorical features based on quantiles
categorical_data = sapply(continuous_data, function(x) (0:2)[cut(x,breaks = 3)])
head(categorical_data)

# Categorical latent class model
model = stepmix(n_components=3, measurement="categorical",
                verbose=1, random_state=123)

# Fit model
cat_fit <- fit(model, categorical_data)

# Class predictions
iris['cat_pred'] = predict(cat_fit, X = categorical_data)

### recode the class to respect the order of iris Species.
iris$cat_pred <- c(0,2,1)[iris$cat_pred + 1]
xtabs(~Species + cat_pred, iris)

fossil::rand.index(as.numeric(iris$Species), iris$cat_pred)

# More complex models need a more complex description
# StepMix provides a function to quickly build mixed descriptors for DataFrames
mixed_data = cbind(continuous_data[, c("Sepal.Length", "Petal.Length")],
                   data.frame(Petal.Width = binary_data[,4]),
                   data.frame(Sepal.Width = categorical_data[,2]))
# Unspecified variables are simply not included in mixed_data
md = mixed_descriptor(data = mixed_data, continuous = 1:2, binary = 3,
     categorical = 4)

### md contains the dataframe and the descriptor
names(md)

# Pass descriptor to StepMix and fit model
model = stepmix(n_components=3, measurement=md$descriptor, random_state=as.integer(123))
# Fit model
mixed_fit <- fit(model, md$data)
# Class predictions
md$data['mixed_pred'] <- predict(mixed_fit, X = md$data)
table(md$data[,'mixed_pred'],iris$Species)

fossil::rand.index(as.numeric(iris$Species), md$data$mixed_pred)

# Initialize the random seed so that the result is the same at
# every run
set.seed(1325)
continuous_data_nan <- as.matrix(continuous_data)
continuous_data_nan[sample(1:600,0.20*600)] <- NaN
continuous_data_nan <- as.data.frame(continuous_data_nan)
continuous_data_nan

# Create a continuous latent class model
model = stepmix(n_components=3, measurement="continuous_nan",
                verbose=1, random_state=123)

# Fit to data
model.fit <- fit(model, continuous_data_nan)

# Save class membership predictions to df
continuous_data_nan[,'continuous_pred_nan'] <- predict(model.fit, continuous_data_nan)
table(continuous_data_nan[,'continuous_pred_nan'], iris$Species)

fossil::rand.index(as.numeric(iris$Species), continuous_data_nan$continuous_pred_nan)

