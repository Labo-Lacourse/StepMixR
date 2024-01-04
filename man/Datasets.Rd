\name{Datasets}
\alias{random_nan}
\alias{bakk_measurements}
\alias{data_bakk_response}
\alias{data_bakk_covariate}
\alias{data_bakk_complete}
\alias{data_generation_gaussian}
\alias{data_gaussian_diag}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Series of function to simulate data.
}
\description{
 These functions generates data with multiple groups using different distributions and optionnaly
 adding a level of missing value.
}
\usage{
random_nan(X, Y, nan_ratio, random_state=NULL)
bakk_measurements(n_classes, n_mm, sep_level)
data_bakk_response(n_samples, sep_level, n_classes = 3, n_mm = 6, random_state = NULL)
data_bakk_covariate(n_samples, sep_level, n_mm = 6, random_state = NULL)
data_bakk_complete(n_samples, sep_level, n_mm=6, random_state=NULL, nan_ratio=0.0)
data_generation_gaussian(n_samples, sep_level, n_mm=6, random_state=NULL)
data_gaussian_diag(n_samples, sep_level, n_mm = 6, random_state = NULL, nan_ratio = 0.0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{The X matrix or data.frame for the measurement part of the model}
  \item{Y}{The Y matrix or data.frame for the structural part of the model}
  \item{nan_ratio}{The ratio of missing values. A value between 0 and 1.}
  \item{random_state}{An integer initializing the seed of the random generator.}
  \item{n_classes}{Number of latent classes required.}
  \item{n_mm}{Number of features in the measurement model.}
  \item{sep_level}{Separation level in the measurement data.}
  \item{n_samples}{Number of samples.}
}
\details{
    These function returns simulated data used to test the package.
}
\value{
   list of data.frame simulated according to the function parameters.
}
\references{
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

\examples{
if (reticulate::py_module_available("stepmix")){
  sim1 <- data_gaussian_diag(n_samples=50, sep_level=0.5, n_mm = 6, nan_ratio = 0.1)
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
