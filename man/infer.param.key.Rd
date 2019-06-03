\name{infer.param.key}

\docType{data}

\alias{infer.param.key}

\title{Initialising values for key parameters inferred with the \code{\link{infer}} function.}

\description{
This data set is an example of the key initialising parameter inputs used in the Bayesian SEIR MCMC that reconstructs the outbreak and infers parameters and missing data.

}

\usage{data(infer.param.key)}

\format{
  A data frame with 1 observation of the following 17 variables:
  \describe{
    \item{alpha}{Initialising value for the background (primary) transmission rate of infection}
    \item{beta}{Initialising value for the secondary transmission rate}
    \item{lat_mu}{Initialising value for the mean of the duration of the farm-level latent period.}
    \item{lat_var}{Initialising value for the variance of the duration of the farm-level latent period.}
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

\examples{
  para.key.inits <- data.frame('alpha' = 2e-4,
                               'beta' = 0.1,
                               'lat_mu' = 5,
                               'lat_var' = 1,
                               'c' = 10,
                               'd' = 3,
                               'k_1' = 3,
                               'mu_1' = 3e-05,
                               'mu_2' = 1e-06,
                               'p_ber' = 0.2,
                               'phi_inf1' = 1,
                               'phi_inf2' = 1,
                               'rho_susc1' = 1,
                               'rho_susc2' = 1,
                               'nu_inf' = 0.2,
                               'tau_susc'= 0.1,
                               'beta_m'= 0.5)
}


\references{
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\keyword{datasets}
