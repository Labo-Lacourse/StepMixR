\name{predict.stepmix.stepmix.StepMix}
\alias{predict.stepmix.stepmix.StepMix}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Predict the membership using the fit of the stepmix python package. 
}
\description{
  Predict the membership of a mixture using a stepmix object in python using X and
  optionally Y to the object.  
}
\usage{
\method{predict}{stepmix.stepmix.StepMix}(object, X = NULL, Y = NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{An object created with the fit function.}
  \item{X}{The X matrix or data.frame for the measurement part of the model}
  \item{Y}{The Y matrix or data.frame for the structural part of the
    model} 
  \item{...}{not used in this function}
}
%% \details{
%% }
\value{
   A vector containing the membership of the mixture.
}
\references{
Bolck, A., Croon, M., and Hagenaars, J. Estimating latent structure
models with categorical variables: One-step versus three-step
estimators. Political analysis, 12(1): 3-27, 2004. 

Vermunt, J. K. Latent class modeling with covariates: Two improved
three-step approaches. Political analysis, 18 (4):450-469, 2010. 

Bakk, Z., Tekle, F. B., and Vermunt, J. K. Estimating the association
between latent class membership and external variables using
bias-adjusted three-step approaches. Sociological Methodology,
43(1):272-311, 2013. 

Bakk, Z. and Kuha, J. Two-step estimation of models between latent
classes and external variables. Psychometrika, 83(4):871-892, 2018 
}
\author{
  Charles-Édouard Giguère (stepmixr)
  Sacha Morin (stepmix)
  Robin Legault (stepmix)
}

%\seealso{
% \code{\link{fit}}, \code{\link{help}} 
%}

\examples{
if (reticulate::py_module_available("stepmix")) {
  require(stepmixr)
  model1 <- stepmix(n_components = 3, n_steps = 2, measurement = "continuous")
  X <- iris[c(1:10, 51:60, 101:110), 1:4]
  fit1 <- fit(model1, X)
  pr1 <- predict(fit1, X)
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
