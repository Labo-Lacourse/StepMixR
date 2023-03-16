\name{mixed_descriptor}
\alias{mixed_descriptor}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Utility function for mixture using mixed description.  
}
\description{
 This function creates a data.frame ordered by continuous, binary and
 categorical columns. It also creates a list used if the model uses
 mixed column types.   
}
\usage{
 mixed_descriptor(data, continuous = NULL, binary = NULL, categorical = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{Data.frame with the mixed data}
  \item{continuous}{index or name of continuous column}
  \item{binary}{index or name of binary column}
  \item{categorical}{index or name of categorical column}
}
\details{
  This methods returns a list of a data.frame sorted by continuous,
  binary and categorical columns. It contains also a descriptor that can
  be used in the measurement section. 
}
\value{
   A list containing data and a descriptor.
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
md <- mixed_descriptor(iris, continuous = 1:4, categorical = 5)
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
