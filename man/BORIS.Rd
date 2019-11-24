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
Simon M. Firestone <simon.firestone@unimelb.edu.au>, Max S.Y. Lau, Saritha Kodikara and Haydar Demirhan, with contributions from Glenn Marion, George Streftaris, Gavin Gibson, Yoko Hayama, Takehisa Yamamoto, Toshi Tsutsui, Lewis Stone and Mark A. Stevenson


Maintainer: Simon Firestone <simon.firestone@unimelb.edu.au>
}
\references{
Firestone, S.M., Hayama, Y., Lau, M.S., Yamamoto, T., Nishi, T., Bradhurst, R.A., Demirhan, H., Stevenson, M.A., Tsutsui, T., 2019. Bayesian transmission network reconstruction for foot-and-mouth disease outbreaks incorporating farm-level covariates. bioRxiv, 835421.

Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.

The current implementation is a wrap-up of the original scripts, with further modifications, available \href{https://github.com/msylau/A-Systematic-Bayesian-Integration-of-Epidemiological-and-Genetic-Data}{here}.


===================================================================

Acknowledgements:

The research that led to this package development was supported by:

1. An Australian Research Council Discovery Early Career Researcher
Award (DE160100477).

2. The Japanese Ministry of Agriculture, Forestry and Fisheries (Management Technologies for the Risk of Introduction of Livestock Infectious Diseases and Their Wildlife-borne Spread in Japan, FY2018-2022).

3. Heriot-Watt University and the Scottish Government's Rural and Environment Science and Analytical Services Division (RESAS).

4. A consultancy agreement undertaken for the New Zealand Ministry of Primary Industries.

5. The University of Melbourne's High Performance Computing system SPARTAN (Lafayette et al., 2016).

===================================================================
}
\keyword{ package }
\seealso{
  \link{infer} \link{sim}
}
