\name{infer.param.sf}

\docType{data}

\alias{infer.param.sf}

\title{Scaling factors for key parameters inferred with the \code{\link{infer}} function.}

\description{
This data set is an example of the scaling factor inputs used in the Bayesian SEIR MCMC that reconstructs the outbreak and infers parameters and missing data.

}

\usage{data(infer.param.sf)}

\format{
  A data frame with 1 observation of the following 17 variables:
  \describe{
    \item{alpha_sf}{Scaling factor for the proposal (i.e. operator) for the background (primary) transmission rate of infection}
    \item{beta_sf}{Scaling factor for the proposal (i.e. operator) for the secondary transmission rate}
    \item{lat_mu_sf}{Scaling factor for the proposal for the mean of the duration of the farm-level latent period.}
    \item{lat_var_sf}{Scaling factor for the proposal for the variance of the duration of the farm-level latent period.}
    \item{c_sf}{Scaling factor for the proposal for the scale parameter of Weibull distribution representing the mean infectious period.}
    \item{d_sf}{Scaling factor for the proposal for the shape parameter of Weibull distribution representing the mean infectious period. \eqn{=}1.0  for an Exponential distribution.}
    \item{k_1_sf}{Scaling factor for the proposal for the spatial transmission kernel shape parameter.}
    \item{mu_1_sf}{Scaling factor for the proposal for the rate of transition mutations based on Kimura's 2 parameter nucleotide substitution model (Kimura, 1980).}
    \item{mu_2_sf}{Scaling factor for the proposal for the rate of transversion mutations based on Kimura's 2 parameter nucleotide substitution model (Kimura, 1980).}
    \item{p_ber_sf}{Scaling factor for the proposal for the probability that a nucleotide base of each of the primary (seeding) sequences has of differing from the base at the corresponding site in the sequence of the universal master sequence [for details see (Lau et al., 2015)].}
    \item{phi_inf1_sf}{Scaling factor for the proposal for the multiplicative effect of predominant species on premises-level infectivity compared to a reference of ftype0 farms, for ftype1 farms.}
    \item{phi_inf2_sf}{Scaling factor for the proposal for the multiplicative effect of predominant species on premises-level infectivity compared to a reference of ftype0 farms, for ftype2 farms.}
    \item{rho_susc1_sf}{Scaling factor for the proposal for the multiplicative effect of predominant species on premises-level susceptibility compared to a reference of ftype0 farms, for ftype1 farms.}
    \item{rho_susc2_sf}{Scaling factor for the proposal for the multiplicative effect of predominant species on premises-level susceptibility compared to a reference of ftype0 farms, for ftype2 farms.}
    \item{nu_inf_sf}{Scaling factor for the proposal for the effect (power) of number of animals on premises-level infectivity for farms.}
    \item{tau_susc_sf}{Scaling factor for the proposal for the effect (power) of number of animals on premises-level susceptibility for farms.}
    \item{beta_m_sf}{Scaling factor for the proposal for the secondary transmission rate by contact-related transmission/animal movement (under development).}    
}
}

\examples{
  para.sf <- data.frame('alpha_sf' = 0.001,
                        'beta_sf' = 0.5,
                        'lat_mu_sf'	= 1.25, 
                        'lat_var_sf' = 1.75,
                        'c_sf' = 1.25,
                        'd_sf' = 0.75,
                        'k_1_sf' = 1,
                        'mu_1_sf' = 2.5e-5,
                        'mu_2_sf' = 2.5e-6,
                        'p_ber_sf' = 0.02,
                        'phi_inf1_sf' = 1.75,
                        'phi_inf2_sf' = 1.5,
                        'rho_susc1_sf' = 1,
                        'rho_susc2_sf' = 1.25,
                        'nu_inf_sf' = 0.25,
                        'tau_susc_sf' = 0.25,
                        'beta_m_sf' = 1)
}


\references{
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\keyword{datasets}
