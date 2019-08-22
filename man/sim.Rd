\name{sim}
\alias{sim}
\title{Simulate an outbreak with genomic data}

\description{
  \code{sim} implements a simulation routine based on Selke's threshold method.
}

\usage{
  sim(epi.inputs = NULL, 
      moves.inputs = NULL, 
      parsKey = NULL,
      parsAux = NULL, 
      inputPath = NULL, 
      outputPath = NULL,
      debug = NULL)
}

\arguments{
  \item{epi.inputs}{A \code{data.frame} including farm covariate data in the format detailed in \code{\link{sim.epi.input}}.}
  \item{moves.inputs}{A \code{data.frame} including animal movement/contact-tracing data in the format detailed in \code{\link{sim.moves.input}}.}
  \item{parsKey}{A \code{data.frame} including key values for parameters that drive the simulation in the format detailed in \code{\link{sim.param.key}}.}
  \item{parsAux}{A \code{data.frame} including important auxillary parameters for the simulation in the format detailed in \code{\link{sim.param.aux}}.}
  \item{inputPath}{The path to the folder where the input files will be created.}
  \item{outputPath}{The path to the folder where the output files will be created.}
  \item{debug}{A logical that produces verbose outputs for debugging only if \eqn{=}1.}  
}

\details{
  
  \code{sim} simulates outbreak datasets with epidemiological and genetic data based on input data following the approach described in Lau (2015) based on Sellke thresholds (Sellke, 1983). 
  
  The \code{sim} function calls compiled C++ source code and works upon inputs from the inputPath and dumps extensive outputs in the outputPath. NOTE: \code{sim} deletes all files in the inputPath and outputPath directories prior to simulation.
  
  See the respective input data examples for the format of required inputs: \code{\link{sim.epi.input}}, \code{\link{sim.moves.input}}, \code{\link{sim.param.key}} and \code{\link{sim.param.aux}}.
  
}

\value{
  This function outputs the following files at \code{outputPath} that are explained below:
    
  \item{con_seq_current.csv}{The grand master sequence, Gm, from which all index sequences are derived. 1,2,3,4 represent nucleotides A,G,T,C, respectively.}
  \item{epi_sim.csv}{The simulated epidemiological dataset in the same format as the epi input data, now including simulated timings for exposure ("\code{t_e}"), onset of infectiousness ("\code{t_i}") and recovery/removal ("\code{t_r}") for each individual.}
  \item{infected_source.txt}{The simulated infectious source of each individual, based on it's C++ index, i.e. source 0 is the first individual in the dataset (k=0).}
  \item{sampled_perct.txt}{The proportion of infected individuals that were sampled.}  
  \item{subject_xx_nt.txt}{Simulated nucleotide data for individual xx. If there are multiple transmission or sampling events for this individual, then each line represents the sequence at the time-point in subject_xx_t_nt.txt}
  \item{subject_xx_t_nt.txt}{Simulated time-points for sequences for individual xx, corresponding to the sequences stored in lines in subject_xx_nt.txt}
  \item{t_sample.txt}{Simulated time-points of sampling events for individual xx. If not sampled then \eqn{=} unassigned value (e.g., 9e06).}
  
}

\references{
  
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., 2015. Novel Bayesian inference in epidemics - model assessment and integrating epidemiological and genetic data. Department of Actuarial Mathematics & Statistics, School of Mathematical and Computer Sciences. Heriot-Watt University.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.

Sellke, T., 1983. On the asymptotic distribution of the size of a stochastic epidemic. Journal of Applied Probability 20, 390-394.

}

\author{
  Simon Firestone, Max S.Y. Lau, Haydar Demirhan, Saritha Kodikara
  
  Maintainer: Simon Firestone <simon.firestone@unimelb.edu.au>
}

\examples{
  data(sim.epi.input)
  data(sim.moves.input)

  # The key simulation parameters                     
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

  # The auxilliary parameters 
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
                         'opt_mov' = 0,
                         'n_mov' = 60,
                         stringsAsFactors = FALSE)  

  

  \dontrun{
    sim.out<-sim(epi.inputs = sim.epi.input, 
                 moves.inputs = sim.moves.input, 
                 parsKey = para.key,
                 parsAux = pars.aux, 
                 inputPath = "./inputs", 
                 outputPath = "./outputs")
  }

# See package vignette for a complete example.
  
}