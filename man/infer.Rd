\name{infer}
\alias{infer}
\title{Implement SEIR MCMC inference based on epidemiological and genomic data}

\description{
\code{infer} implements the SEIR MCMC model to infer the transmission network and key epidemiological and phylogenetic parameters.
}

\usage{
  infer(covariates = NULL,
        moves.inputs = NULL,
        parsAux = NULL,
        keyInits = NULL,
        priors = NULL,
        scalingFactors = NULL,
        seed = NULL,
        accTable = NULL,
        t.sample = NULL,
        inputPath = NULL,
        outputPath = NULL,
        dnaPath = NULL,
        debug = NULL)
}

\arguments{
  \item{covariates}{A \code{data.frame} including farm covariate data in the format detailed in \code{\link{infer.epi.input}}.}
  \item{moves.inputs}{A \code{data.frame} including animal movement/contact-tracing data in the format detailed in \code{\link{infer.moves.input}}.}
  \item{parsAux}{A \code{data.frame} including the parameters for MCMC implementation in the format detailed in \code{\link{infer.param.aux}}.}
  \item{keyInits}{A \code{data.frame} including key initial values for parameters. For definition of elements see \code{\link{infer.param.key}}.}
  \item{priors}{A \code{data.frame} including prior settings for parameters. For definition of elements see \code{\link{infer.param.priors}}.}
  \item{scalingFactors}{A \code{data.frame} including scaling factors for paramaters, also known as operators or proposal distances. For definition of elements see \code{\link{infer.param.sf}}.}
  \item{seed}{An \code{integer} seed for for the random number generator.}
  \item{accTable}{A \code{data.frame} including known the sources of infection for each individual for simulated outbreaks, or \code{9999} for each unknown source.}
  \item{t.sample}{A \code{data.frame} of the timing of sampling that led to genomic data being available for each individual. If not sampled, then enter the \code{unassigned_value}.}  
  \item{inputPath}{The path to the folder where the input files will be created.}
  \item{outputPath}{The path to the folder where the output files will be created.}
  \item{dnaPath}{The path to sequence data. This full path must point to a directory containing a single \code{.fasta} file.}
  \item{debug}{A logical that produces verbose outputs for debugging only if \eqn{=}1.}

}

\details{

\code{infer} implements the Susceptible Exposed Infectious Recovered (SEIR) systematic Bayesian Markov Chain Monte Carlo (MCMC) inference as described in Firestone et. al. (under peer review) which is an extension of that originally presented by Lau et. al. (2015), incorporating additional terms representing individual covariates (risk categories) in the epidemiological data. 

The \code{infer} function calls compiled C++ source code and works upon inputs from the inputPath and dumps extensive outputs in the outputPath. NOTE: \code{infer} deletes all files in the inputPath and outputPath directories prior to simulation.
}

\value{
  This function outputs six files at \code{outputPath} that are explained below:
 
\item{con_seq_current.csv}{The grand master sequence, Gm, output each \code{n_output_gm} iterations (one line per sequence) as a comma-separated file with the nucleotides A,G,T,C, represented by 1,2,3,4, respectively.}
\item{infected_source_current.csv}{The inferred sources of each individual output each \code{n_output_source} iterations (one line per iteration) as a comma-separated file.}
\item{parameters_current.log}{The inferred parameters at each step of the MCMC chain (one line per iteration), with the first 10 discarded so that it can be opened directly in the program \href{http://beast.community/tracer}{Tracer}.}
\item{seqs_current.csv}{The inferred sequence for each individual, output each \code{n_output_gm} iterations (one line per sequence), as a comma-separated file with the nucleotides A,G,T,C, represented by 1,2,3,4, respectively. The inferred sequences for the first individual can be found in the rows: \code{seq(1,n_iterations,by=n_output_gm)}.
}
\item{t_e_current.csv}{Inferred timing of exposure for each infected individual (one line per iteration) as a comma-separated file.}
\item{t_i_current.csv}{Inferred timing of onset of infectiousness for each infected individual (one line per iteration) as a comma-separated file.}
}

\references{

Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.

Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\author{
Simon Firestone, Max S.Y. Lau, Haydar Demirhan

Maintainer: Simon Firestone <simon.firestone@unimelb.edu.au>
}

\examples{
 data(infer.epi.input)
 data(infer.moves.input)
 
 n<-nrow(infer.epi.input)
 
 
  # The auxilliary parameters 
  pars.aux <- data.frame('n' = 100,
                         'kernel_type' = 'power_law',
                         'coord_type' = 'longlat',
                         't_max' = 100,
                         'unassigned_time' =  9e+6,
                         'processes' = 1,
                         'n_seq' = 5,                       
                         'n_base' = 7667,                       
                         'n_iterations' = 1e5,
                         'n_frequ' = 10,
                         'n_output_source' = 1,
                         'n_output_gm' = 1000,
                         'n_cout' = 10,
                         'opt_latgamma' = 1,
                         'opt_k80' = 1,
                         'opt_betaij' = 1,
                         'opt_ti_update' = 1,
                         'opt_movt' = 0,
                         stringsAsFactors = FALSE)
 
 # Initial values for the key parameters                     
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
                      
 # The prior information                      
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

  #scaling factors (also known as operators or proposal distances)
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



\dontrun{
  infer.out<-infer(covariates = infer.epi.input,
                   moves.inputs = infer.moves.input,
                   parsAux = pars.aux, 
                   keyInits = para.key.inits,
                   priors = para.priors, 
                   scalingFactors = para.sf, 
                   seed = 1,
                   accTable = data.frame(rep(9999, n)),
                   t.sample = infer.epi.input$t_s,
                   inputPath = "./inputs", 
                   outputPath = "./outputs", 
                   dnaPath = "./gen_inputs")                   
}

# See package vignette for a complete example.

}