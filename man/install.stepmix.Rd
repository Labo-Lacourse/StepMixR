\name{install.stepmix}
\alias{install.stepmix}
\alias{check_pystepmix_version}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Install stepmix python package into python via reticulate. 
}
\description{
  Install the stepmix python package in the python instance used by reticulate.
}
\usage{
install.stepmix(envname, method, conda, pip, ...)
check_pystepmix_version()
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{envname}{Name of the python environment. "r-reticulate" by
    default.}
  \item{method}{installation method. See doc in reticulate}
  \item{conda}{Path to a conda install. See doc in reticulate}
  \item{pip}{Logical value to choose pip as the install method}
  \item{...}{Not used in function}
}
\details{
  This methods installs stepmix in the python instance or environment
  used by reticulate. It uses reticulate::py_install.  
}
\value{
   It doesn't return anything.
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

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
