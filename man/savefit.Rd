\name{savefit}
\alias{savefit}
\alias{loadfit}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Save the fit of a mixture using the stepmix python package. 
}
\description{
  This function saves the stepmix fitted object in python using the
  pickle package. 
}
\usage{
savefit(fitx, f)
loadfit(f)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{fitx}{An object created with the stepmix function.}
  \item{f}{String indicating the name of the file}
}
\details{
  This methods allows to save/load the stepmix object in a binary file using
  the pickle package.  
}
\value{
   A pointer to a python object of type StepMix.
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
  model1 <- stepmix(n_components = 2, n_steps = 3)
  X <- data.frame(x1 = c(0,1,1,1,1,0,0,0,0,0,1,1,0),
                  x2 = c(0,1,1,0,0,1,1,0,0,0,1,0,1))
  fit1 <- fit(model1, X)
  savefit(fit1, "fit1.pickle")
  
  ### clean the directory.
  file.remove("fit1.pickle")
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
