### predict stepmixr.fit

devtools::install_github("giguerch/stepmixr")

require(stepmixr)
reticulate::use_python("C:/Python39/")

mod1 <- stepmix()
X = data.frame(x1 = c(0,1,0,1,1,0,0,0,1,1,1,0,0,1,0,1,1,0,0,1),
               x2 = c(0,1,0,1,1,0,0,0,1,0,1,1,1,1,0,1,0,0,0,1))
fit1 <- fit(mod1, X)

pr1 <- predict(fit1, X)
pr1

savefit(fit1, "test_fit1.pickle")

### reload(R)

require(stepmixr)
reticulate::use_python("C:/Python39/")

fit1 <- loadfit("test_fit1.pickle")
fit1$get_parameters()

X <- iris[,3:4]
mod2 <- stepmix(n_components = 3, measurement = "gaussian_diag",
                random_state = 1234, verbose = 1)
fit2 <- fit(mod2, X)
pr2 <- predict(fit2, X)

table(c(2,1,3)[pr2+1], iris$Species)
fit2$get_parameters()
help(stepmix)

### Ajouter sample weight comme option à fit.


require(stepmixr)
#reticulate::use_python("C:/python39")
mod <- stepmix(n_components = 3, n_steps = 3, measurement = "gaussian_unit",
               random_state = 1235)
fit1 <- fit(mod, as.matrix(iris[,3:4]))
fit1$get_params()
fit1$predict(X= iris[,3:4])
fit1$get_parameters()
predict(fit1, X = iris[,3:4])
table(predict(fit1, X = iris[,3:4]), iris$Species)


require(stepmixr)
continuous_data = iris[,1:4]
continuous_data
model <- stepmix(n_components=3, measurement="continuous", verbose=1, 
                 random_state=123)
# Fit to data
fit <- fit(model, continuous_data)
