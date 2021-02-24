\name{infer.param.priors}

\docType{data}

\alias{infer.param.priors}

\title{The prior information for parameters inferred with the \code{\link{infer}} function.}

\description{
This data set is an example of the prior parameterisation used in the Bayesian SEIR MCMC that reconstructs the outbreak and infers parameters and missing data.

}

\usage{data(infer.param.priors)}

\format{
  A data frame with 1 observation of the following 27 variables:
  \describe{
    \item{t_range}{Used for update of the time of onset of infectiousness (t_i) if opt_ti_update\eqn{=}1, proposed values of t_i range between time of onset of clinical signs (t_o) \eqn{+-} range (for details see paper Supporting Information in Lau et al., 2015).}
    \item{t_back}{Maximum assumed length for the latent period.}
    \item{t_bound_hi}{Used when proposing initial sources, t_i of infectee must be within t_bound_hi days of proposed t_i of j (essentially the upper bound of the generation interval).}
    \item{rate_exp_prior}{The rate of exposure to be used as the mode of a vague prior.}
    \item{ind_n_base_part}{Logical representing whether the genomic sequence data is partial (\eqn{=}1) or complete (\eqn{=}0).}
    \item{n_base_part}{The partial sequence length, used if ind_n_base_part\eqn{=}1.}
    \item{alpha_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{beta_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{mu_lat_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{var_lat_lo}{Lower/upper bound for prior of inferred parameter.}
    \item{var_lat_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{c_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{d_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{k_1_lo}{Lower/upper bound for prior of inferred parameter.}
    \item{k_1_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{mu_1_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{mu_2_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{p_ber_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{phi_inf1_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{phi_inf2_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{rho_susc1_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{rho_susc2_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{nu_inf_lo}{Lower/upper bound for prior of inferred parameter.}
    \item{nu_inf_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{tau_susc_lo}{Lower/upper bound for prior of inferred parameter.}
    \item{tau_susc_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{beta_m_hi}{Lower/upper bound for prior of inferred parameter.}
    \item{trace_window}{Under development: The maximum time that a movement could have occurred earlier than exposure to be considered causally related to infection.}
}
}

\examples{
 para.priors <- data.frame('t_range' = 7,
                          't_back' = 21,
                          't_bound_hi' = 10,
                          'rate_exp_prior' = 0.001,
                          'ind_n_base_part' = 0,
                          'n_base_part' = 1000,
                          'alpha_hi' = 0.1,
                          'beta_hi' = 50,
                          'mu_lat_hi' = 50,
                          'var_lat_lo' = 0.1,
                          'var_lat_hi' = 50,
                          'c_hi' = 100,
                          'd_hi' = 100,
                          'k_1_lo' = 0,                          
                          'k_1_hi' = 100,
                          'mu_1_hi' = 0.1,
                          'mu_2_hi' = 0.1,
                          'p_ber_hi' = 1.0,
                          'phi_inf1_hi' = 500,
                          'phi_inf2_hi' = 500,
                          'rho_susc1_hi' = 500,
                          'rho_susc2_hi' = 500,
                          'nu_inf_lo' = 0,
                          'nu_inf_hi' = 1,
                          'tau_susc_lo' = 0,
                          'tau_susc_hi' = 1,
                          'beta_m_hi' = 5,
                          'trace_window' = 20) 
}


\references{
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\keyword{datasets}
