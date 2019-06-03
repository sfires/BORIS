\name{sim.param.aux}

\docType{data}

\alias{sim.param.aux}

\title{Auxillary input parameters for simulating outbreaks with the \code{\link{sim}} function.}

\description{
This data set is an example of the additional parameter inputs that shape the outbreak that will be simulated, along with corresponding genomic data.
}

\usage{data(sim.param.aux)}

\format{
  A data frame with 1 observations of the following 17 variables:
  \describe{
    \item{n}{number of individual farms at risk}
    \item{seed}{Seed for the random number generator to enable simulation runs to be reproducible. }
    \item{n_base}{Number of bases (nucleotides) in length for the sequences to be simulated with the Kimura (1980) model. Presently limited to 10,000. For longer sequences, it is suggested to simulate the partial sequence of SNPs only.}
    \item{n_seq}{Initial number of sequences expected per indivual unit at risk. Defines memory allocated, updated if more are needed.}
    \item{t_max}{Upper limit of observation period.}
    \item{unassigned_time}{An abritary extreme value e.g. 9e+10, to indicate that an event does not happen e.g. no infection.}
    \item{sample_range}{The maximum possible delay between infection and sampling. Realistic upper bounds for some infectious diseases with overt clinical signs in naive populations can be assumed to be between 14-21 days.}
    \item{partial_seq_out}{Logical representing whether the genomic sequence data to be simulated is partial (\eqn{=}1) or complete (\eqn{=}0).}
    \item{n_base_part}{The partial sequence length, used if partial_seq_out\eqn{=}1.}
    \item{n_index}{The number of indexes to be simulated. For a transmission tree with multifurcation use \eqn{ge}1.}
    \item{coord_type}{Coordinate system: decimal degrees (i.e. latitudes and longitudes) or cartesian (projected coordinate reference system) can take the values "\code{longlat}" / "\code{cartesian}"}
    \item{kernel_type}{Kernel type (string variable) that can take the values "\code{power_law}" / "\code{exponential}" / "\code{cauchy}" / "\code{gaussian}".}
    \item{latent_type}{Distribution type used for simulating the latent period. Presently only implemented for "\code{gamma}"}
    \item{opt_k80}{Implementation option: 1 \eqn{=} reformulated K80 DNA substitution model to match original 1980 paper (0 \eqn{=} original version in Lau et al. (2015) based on a secondary reference).}
    \item{opt_betaij}{Implementation option: 1 \eqn{=} farm-level covariates incorporated into beta, i.e. \eqn{\beta_{ij}} Lau modified model from Firestone et al. (2019) (0 \eqn{=} originally implemented model from Lau et al. (2015).}
    \item{opt_movt}{Implementation option (under development): 1 \eqn{=} utilise animal movement/contact-tracing data in the simulation (0 \eqn{=} Lau et al. (2015) original implementation)
}
    \item{n_mov}{number of animal movements or contacts in the inputted data \code{\link{sim.moves.input}}}
  }
}

\examples{
  pars.aux <- data.frame('n' = 100,
                         'seed' = 2468,
                         'n_base' = 7667,
                         'n_seq' = 5,
                         't_max' = 100,
                         'unassigned_time' =  9e+6,
                         'sample_range' = 10,
                         'partial_seq_out' = 0,
                         'n_base_part' = 1000,
                         'n_index' = 1,
                         'coord_type' = 'longlat',                       
                         'kernel_type' = 'power_law',
                         'latent_type' = 'gamma',
                         'opt_k80' = 1,
                         'opt_betaij' = 1,
                         'opt_movt' = 0,
                         'n_mov' = 60,
                         stringsAsFactors = FALSE) 
}


\references{
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\keyword{datasets}
