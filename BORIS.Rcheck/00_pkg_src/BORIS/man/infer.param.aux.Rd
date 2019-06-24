\name{infer.param.aux}

\docType{data}

\alias{infer.param.aux}

\title{Auxillary input parameters for configuring the MCMC used by the \code{\link{infer}} function.}

\description{
This data set is an example of the additional parameter inputs that configure the Bayesian SEIR MCMC that reconstructs the outbreak and infers parameters and missing data.
}

\usage{data(infer.param.aux)}

\format{
  A data frame with 1 observations of the following 18 variables:
  \describe{
    \item{n}{Number of individuals/farms, presently limited to 500 considering RAM requirements.}
    \item{kernel_type}{Kernel type (string variable) that can take the values "\code{power_law}" / "\code{exponential}" / "\code{cauchy}" / "\code{gaussian}".}
    \item{coord_type}{Coordinate system: decimal degrees (i.e. latitudes and longitudes) or cartesian (projected coordinate reference system) can take the values "\code{longlat}" / "\code{cartesian}"}
    \item{t_max}{Upper limit of observation period.}
    \item{unassigned_time}{An abritary extreme value e.g. 9e+10, to indicate that an event does not happen e.g. no infection.}
    \item{processes}{Number of parallel processes (if run on a single computer \eqn{=}1; only set to \eqn{\ge} 1 if on a SLURM  cluster and submitted with a task array for parallel computation of multiple chains)}
    \item{n_seq}{Initial number of sequences expected per indivual unit at risk. Defines memory allocated, defaults to 5, updated if more are needed.}
    \item{n_base}{Number of bases (nucleotides) in length for the sequences to be simulated with the Kimura (1980) model. Presently limited to 10,000. For longer sequences, it is suggested to use partial sequences of SNPs only.}
    \item{n_iterations}{Number of iterations to run the MCMC. Typically 100,000s to millions of iterations are required to reach convergence of multiple chains, which should be checked. An additional 10 iterations are included, so that the first 10 can be discarded from the output file \code{parameters_current.log} and it can then be opened directly in the program \href{http://beast.community/tracer}{Tracer}.}
    \item{n_frequ}{Frequency of updating exposure times.}
    \item{n_output_source}{Frequency of outputing updated sources for each known infected farm.}
    \item{n_output_gm}{Frequency of outputting an update of the grand master sequence (Gm) and the inferred sequences of each individual.}
    \item{n_cout}{Frequency of updating the console output.}
    \item{opt_latgamma}{Implementation option: 1 \eqn{=} assume a Gamma distribution for the latent period (0 \eqn{=} assume a Gaussian distribution).}
    \item{opt_k80}{Implementation option: 1 \eqn{=} reformulated K80 DNA substitution model to match original 1980 paper (0 \eqn{=} original version in Lau et al. (2015) based on a secondary reference).}
    \item{opt_betaij}{Implementation option: 1 \eqn{=} farm-level covariates incorporated into beta, i.e. \eqn{\beta_{ij}} Lau modified model from Firestone et al. (2019b) (0 \eqn{=} originally implemented model from Lau et al. (2015).}
    \item{opt_ti_update}{Implementation option: 1 \eqn{=} update timing of inferred onset of infectivity (0 \eqn{=} in Lau original implementation based on simulated data, see Supporting Information in Lau et al. (2015)).}
  \item{opt_movt}{Implementation option under development: 1 \eqn{=} animal movements/contact-tracing incorporated into likelihood estimation with inferred parameter \eqn{\beta_m} (0 \eqn{=} originally implemented model from Lau et al. (2015).}
}
}

\examples{
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
}


\references{
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\keyword{datasets}
