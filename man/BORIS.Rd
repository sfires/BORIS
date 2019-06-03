\name{BORIS}
\alias{BORIS}
\docType{package}
\title{
  Bayesian Outbreak Reconstruction Inference and Simulation (BORIS)
}
\description{
  Outbreak reconstruction using epidemiological and genomic data based on the Susceptible Exposed Infectious Recovered (SEIR) systematic Bayesian Markov Chain Monte Carlo (MCMC) algorithm described in Firestone et. al. (under peer review), which extends the model of Lau et. al. (2015). Infers the transmission network, genomes and unobserved parameters at all important time points in the outbreak. Includes functions for forwards simulation of outbreaks using a related algorithm, both for model verification and for Bayesian real-time predictions.
}
\details{
  Key function are compiled from C++ source code, using the Boost header library (requiring 'BH' and 'Rcpp') and work upon inputs from the inputPath and dump extensive outputs in the outputPath, and include:
  \itemize{
  \item \code{sim} for simulating outbreak data 
  \item \code{infer} for infering the transmission network and unobserved parameters.
  }
  
  For an example of usage see the package vignette.
  
  To install from source, requires Rtools to be installed as described in the installation vignette and also \href{http://jtleek.com/modules/01_DataScientistToolbox/02_10_rtools/}{here}.
}
\author{
Simon M Firestone <simon.firestone@unimelb.edu.au>, Max S.Y. Lau, Saritha Kodikara and Haydar Demirhan, with contributions from Glenn Marion, George Streftaris, Gavin Gibson, Yoko Hayama, Takehisa Yamamoto, Toshi Tsutsui, Lewis Stone and Mark Stevenson


Maintainer: Simon Firestone <simon.firestone@unimelb.edu.au>
}
\references{
Firestone, S.M., Hayama, Y., Lau, M.S., Yamamoto, T., Nishi, T., Bradhurst, R.A., Demirhan, H., Stevenson, M.A., Tsutsui, T., under peer review. Bayesian transmission network reconstruction for foot-and-mouth disease outbreaks incorporating farm-level covariates. Proceedings of the Royal Society B: Biological Sciences.

Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}
\keyword{ package }
\seealso{
  \link{infer} \link{sim}
}
