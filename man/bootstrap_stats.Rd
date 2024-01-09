\name{bootstrap_stats}
\alias{bootstrap_stats}
\alias{bootstrap_stats.stepmix.stepmix.StepMix}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Non-parametric boostrap of StepMix estimator. 
}
\description{
Non-parametric boostrap of StepMix estimator. Obtain boostrapped
parameters and some statistics (mean and standard deviation). If a
covariate model is used in the structural model, the output keys
"cw_mean" and "cw_std" are omitted. 
}
\usage{
\method{bootstrap_stats}{stepmix.stepmix.StepMix}(x, X = NULL, y = NULL, n_repetitions = 10, ...)
bootstrap_stats(x, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{An object created with the fit function}
  \item{X}{The X matrix or data.frame for the measurement part of the model}
  \item{y}{The y matrix or data.frame for the structural part of the model}   
  \item{n_repetitions}{The number of bootsrap sample}
  \item{...}{for future options. Currently not used}
}
\details{
  This methods returns a list with bootstrap samples (\code{samples})
  and the log-likelihood (\code{rep_stats}). Mean and standard deviation
  are added to the results.
}
\value{
   A list containing bootstrap samples of the parameters. The mean and
   standard of class weights (\code{cw_mean}, \code{cw_std}),
   measurement model parameters (\code{mm_mean}, \code{mm_std}),
   structural model parameters (\code{sm_mean}, \code{sm_std}) are also
   added.  If a covariate model is used in the structural model, the
   output keys \code{cw_mean} and \code{cw_std} are omitted.
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
  Éric Lacourse, 
  Roxane de la Sablonnière,
  Charles-Édouard Giguère,
  Sacha Morin,
  Robin Legault,
  Félix Laliberté,
  Zsusza Bakk
}

%\seealso{
% \code{\link{fit}}, \code{\link{help}} 
%}

%\examples{
%if (reticulate::py_module_available("stepmix")) {
%  model1 <- stepmix(n_components = 3, n_steps = 2, measurement = "continuous")
%  X <- iris[c(1:10, 51:60, 101:110), 1:4]
%  fit1 <- fit(model1, X)
%  fit1.bs <- bootstrap_stats(fit1, X, nrep)
%}
%}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
