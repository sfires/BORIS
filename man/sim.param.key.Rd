\name{sim.param.key}

\docType{data}

\alias{sim.param.key}

\title{Key input parameters for simulating outbreaks with the \code{\link{sim}} function.}

\description{
This data set is an example of the key parameter inputs that shape the outbreak that will be simulated, along with corresponding genomic data.
}

\usage{data(sim.param.key)}

\format{
  A data frame with 1 observation of the following 17 variables:
  \describe{
    \item{alpha}{The background (primary) transmission rate of infection}
    \item{beta}{The secondary transmission rate}
    \item{mu_1}{The rate of transition mutations based on Kimura's 2 parameter nucleotide substitution model (Kimura, 1980).}
    \item{mu_2}{The rate of transversion mutations based on Kimura's 2 parameter nucleotide substitution model (Kimura, 1980).}
    \item{a}{Shape parameter of Gamma distribution representing the latent period.}
    \item{b}{Scale parameter of Gamma distribution representing the latent period.}
    \item{c}{Scale parameter of Weibull distribution representing the mean infectious period.}
    \item{d}{Shape parameter of Weibull distribution representing the mean infectious period. \eqn{=}1.0  for an Exponential distribution.}
    \item{k_1}{The spatial transmission kernel shape parameter.}
    \item{p_ber}{The  probability that a nucleotide base of each of the primary (seeding) sequences has of differing from the base at the corresponding site in the sequence of the universal master sequence [for details see (Lau et al., 2015)].}
    \item{phi_inf1}{The multiplicative effect of predominant species on premises-level infectivity compared to a reference of ftype0 farms, for ftype1 farms.}
    \item{phi_inf2}{The multiplicative effect of predominant species on premises-level infectivity compared to a reference of ftype0 farms, for ftype2 farms.}
    \item{rho_susc1}{The multiplicative effect of predominant species on premises-level susceptibility compared to a reference of ftype0 farms, for ftype1 farms.}
    \item{rho_susc2}{The multiplicative effect of predominant species on premises-level susceptibility compared to a reference of ftype0 farms, for ftype2 farms.}
    \item{nu_inf}{The effect (power) of number of animals on premises-level infectivity for farms.}
    \item{tau_susc}{The effect (power) of number of animals on premises-level susceptibility for farms.}
    \item{beta_m}{the secondary transmission rate by contact-related transmission/animal movement (under development).}
}
}

\examples{
  para.key <- data.frame('alpha' = 4e-4,
                         'beta' = 0.2,
                         'mu_1' = 2e-05,
                         'mu_2' = 2e-06,
                         'a' = 3.0,
                         'b' = 2.5,
                         'c' = 21.0,
                         'd' = 4.0,
                         'k_1' = 1.7,
                         'p_ber' = 0.1,
                         'phi_inf1' = 3,
                         'phi_inf2' = 1.5,
                         'rho_susc1' = 0.4,
                         'rho_susc2' = 2,
                         'nu_inf' = 0.2,
                         'tau_susc'= 0.1,
                         'beta_m'= 1.0)                  
}


\references{
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\keyword{datasets}
