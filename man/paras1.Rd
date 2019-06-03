\name{paras1}

\docType{data}

\alias{paras1}

\title{Example \code{parameters_current.log} output from \code{infer()}.}

\description{
This data set provides an example of parameter output from the file \code{parameters_current.log} from a single run of \code{infer()}, seeded with 1. Used as an example dataset in the vignette.
}

\usage{data(paras1)}

\format{
  A data frame with 10000 inferred observations (1 per iteration) of the following 20 variables:
  \describe{
    \item{sample}{Iteration number}
    \item{log_likelihood}{Log likelihood as estimated at this iteration}     \item{corr}{Number of individuals for whom the inferred source in this iteration is equal to that in the input argument \code{accTable} to the function \code{infer}}
    \item{alpha}{Initialising value for the background (primary) transmission rate of infection}
    \item{beta}{Initialising value for the secondary transmission rate}
    \item{lat_mu}{Initialising value for the mean of the duration of the farm-level latent period.}
    \item{lat_sd}{Initialising value for the SD of the duration of the farm-level latent period.}
    \item{c}{Initialising value for the scale parameter of Weibull distribution representing the mean infectious period.}
    \item{d}{Initialising value for the shape parameter of Weibull distribution representing the mean infectious period. \eqn{=}1.0  for an Exponential distribution.}
    \item{k_1}{Initialising value for the spatial transmission kernel shape parameter.}
    \item{mu_1}{Initialising value for the rate of transition mutations based on Kimura's 2 parameter nucleotide substitution model (Kimura, 1980).}
    \item{mu_2}{Initialising value for the rate of transversion mutations based on Kimura's 2 parameter nucleotide substitution model (Kimura, 1980).}
    \item{p_ber}{Initialising value for the probability that a nucleotide base of each of the primary (seeding) sequences has of differing from the base at the corresponding site in the sequence of the universal master sequence [for details see (Lau et al., 2015)].}
    \item{phi_inf1}{Initialising value for the multiplicative effect of predominant species on premises-level infectivity compared to a reference of ftype0 farms, for ftype1 farms.}
    \item{phi_inf2}{Initialising value for the multiplicative effect of predominant species on premises-level infectivity compared to a reference of ftype0 farms, for ftype2 farms.}
    \item{rho_susc1}{Initialising value for the multiplicative effect of predominant species on premises-level susceptibility compared to a reference of ftype0 farms, for ftype1 farms.}
    \item{rho_susc2}{Initialising value for the multiplicative effect of predominant species on premises-level susceptibility compared to a reference of ftype0 farms, for ftype2 farms.}
    \item{nu_inf}{Initialising value for the effect (power) of number of animals on premises-level infectivity for farms.}
    \item{tau_susc}{Initialising value for the effect (power) of number of animals on premises-level susceptibility for farms.}
    \item{beta_m}{Initialising value for the secondary transmission rate by contact-related transmission/animal movement (under development).}

  }
}

\references{
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\keyword{datasets}
