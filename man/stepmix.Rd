\name{stepmix}
\alias{stepmix}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
R interface to stepmix in StepMix python. 
}
\description{
  This function creates a basic R list that will be used to initialize the
  stepmix object in python, in order to use the fit and predict function.
}
\usage{
stepmix(n_components = 2, n_steps = 1, 
        measurement = "bernoulli", structural = "bernoulli",
        assignment = "modal", correction = NULL, 
        abs_tol = 1e-3, rel_tol = 1e-10, max_iter = 100,
        n_init = 1, init_params = "random", random_state = NULL,
        verbose = 0, verbose_interval = 10, 
        measurement_params = NULL, structural_params = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{n_components}{The number of latent class. 2 by default.}
  \item{n_steps}{1, 2, or 3, 1 by default. 
    Number of steps in the estimation. Must be one of :
     1: run EM on both the measurement and structural models.
    
     2: first run EM on the measurement model, then on the complete
     model, but keep the measurement parameters fixed for the second
     step. See Bakk, 2018. 

     3: first run EM on the measurement model, assign class
     probabilities, then fit the structural model via maximum
     likelihood. See the correction parameter for bias correction.
     
     See Bakk & Kuha (2018) for more details.}
   \item{measurement}{String describing the measurement model. See
     details for the different available model. The default model
     is "bernouilli"}
   \item{structural}{String describing the structural model. See
     details for the different available model. The default model
     is "bernouilli"}
   \item{assignment}{String indicating the type of class assignments for
     3-step estimation, "modal" by default. Must be one of: 

     soft: keep class responsibilities (posterior probabilities) as is.
    
     modal: assign 1 to the class with max probability, 0 otherwise
     (one-hot encoding).}
   \item{correction}{Bias correction for 3-step estimation. Must be one of :

     None: No correction. Run Naive 3-step.
     
     BCH: Apply the empirical BCH correction from Vermunt, 2004.
     
     ML: Apply the ML correction from Vermunt, 2010, Bakk et al., 2013.
   }
   \item{abs_tol}{The convergence threshold. EM iterations will stop
     when the lower bound average gain is below this threshold. The
     default value is 1e-3.}
   \item{rel_tol}{The convergence threshold. EM iterations will stop
     when the relative lower bound average gain is below this threshold.}

   \item{max_iter}{The number of EM iterations to perform.}

   \item{n_init}{The number of initializations to perform. The best
     results are kept.} 

   \item{init_params}{"kmeans", or  "random", default="random". The
     method used to initialize the weights, the means and the
     precisions. Must be one of: 
     
     kmeans : responsibilities are initialized using kmeans.
     
     random : responsibilities are initialized randomly.}
   
   \item{random_state}{State instance or NULL, default=NULL. Controls the
     random seed given to the method chosen to initialize the
     parameters. Pass an int  for reproducible output across multiple
     function calls.}
   
   \item{verbose}{Default=0. Enable verbose output. If 1, will print
     detailed report of the model and the performance metrics after
     fitting.}

   \item{verbose_interval}{Default=10. Number of iteration done before
     the next print. TODO: Not currently implemented.}

   \item{measurement_params}{Default=NULL, Additional params passed to
     the measurement model class. Particularly useful to specify
     optimization parameters for
     stepmix.emission.covariate.Covariate. Ignored if the measurement
     descriptor is a nested object (see stepmix.emission.nested.Nested).}
   
   \item{structural_params}{Default=NULL, Additional params passed to
     the structural model class. Particularly useful to specify
     optimization parameters for
     stepmix.emission.covariate.Covariate. Ignored if the structural
     descriptor is a nested object (see stepmix.emission.nested.Nested).}
   
}
\details{
  The options for both the measurement and structural part are describe
  here:


  bernoulli: The observed data consists of n_features bernoulli
  (binary) random variables.
  
  bernoulli_nan: the observed data consists of n_features bernoulli
  (binary) random variables. Supports missing values. 

  binary: alias for bernoulli.

  binary_nan: alias for bernoulli_nan.

  categorical: alias for multinoulli.

  categorical_nan: alias for multinoulli_nan.

  continuous: alias for gaussian diag.

  continuous_nan: alias for gaussian_diag_nan. supports missing values.
  
  covariate: covariate model where class probabilities are a multinomial
  logistic model of the features. 

  gaussian: alias for gaussian_unit.
  
  gaussian_nan: alias for gaussian_unit. Supports missing values.

  gaussian_unit: each gaussian component has unit variance. Only fit the mean.
  
  gaussian_unit_nan: each gaussian component has unit variance. Only fit
  the mean. Supports missing values. 

  gaussian_spherical: each gaussian component has its own single variance.

  gaussian_spherical_nan: each gaussian component has its own single
  variance. Supports missing values. 

  gaussian_tied: all gaussian components share the same general covariance matrix.

  gaussian_diag: each gaussian component has its own diagonal covariance matrix.

  gaussian_diag_nan: each gaussian component has its own diagonal
  covariance matrix. Supports missing values. 

  gaussian_full: each gaussian component has its own general covariance matrix.

  multinoulli: the observed data consists of n_features multinoulli
  (categorical) random variables. 

  multinoulli_nan: the observed data consists of n_features multinoulli
  (categorical) random variables. Supports missing values. 
}
\value{
  It returns a list of type stepmixr that contains the arguments of the
  object. 
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

\seealso{
  \code{\link{fit}}
}

\examples{

model1 <- stepmix(n_components = 2, n_steps = 3) 

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
