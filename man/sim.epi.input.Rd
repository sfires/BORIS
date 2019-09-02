\name{sim.epi.input}

\docType{data}

\alias{sim.epi.input}

\title{Epidemiological (covariate) input data on farms in the population to be simulated.}

\description{
This data set provides an example of the inputs on the population of farms housing individual animals at risk in which an outbreak will be simulated, along with corresponding genomic data.
}

\usage{data(sim.epi.input)}

\format{
  A data frame with 100 observations (1 per farm) on the following 11 variables:
  \describe{
    \item{k}{A unique identifier for each farm. Starts at zero given C++ indexing system.}
    \item{coor_x}{x coordinate of the farm's centroid, e.g., Longitude in decimal degrees using WGS84 datum.}
    \item{coor_y}{y coordinate of the farm's centroid, e.g., Latitude in decimal degrees using WGS84 datum.}
    \item{t_e}{Day of exposure after some arbitrary origin (Day 0). Can be set as 0 for the earliest source. Multiple sources are allowed. If left as the unassigned value (default \code{9e06}), will be simulated for individuals that are infected.}
    \item{t_i}{Day when infectious period commences (after Day 0). By definition this is unobserved, so given the unassigned value (default \code{9e06}) and simulated for individuals/farms that are infected and become infectious before \code{t_max}.}
    \item{t_r}{Day when infectious period ends, i.e. recovery or removal of all infectious animals on this farm (possibly through recovery, death or culling). If left as the unassigned value (default \code{9e06}), will be simulated for individuals that become infectious before \code{t_max}.}
    \item{ftype0}{Binary indicator variable for a 3-level categorical variable, used here as \emph{the reference category}, to represent farm type in terms of the predominant susceptible species held, i.e. ftype0 == 1 if predominant species is cattle, otherwise ftype0 == 0.}
    \item{ftype1}{Binary indicator variable used here to represent farm type in terms of the predominant susceptible species held, i.e. ftype1 == 1 if predominant species is pigs, otherwise ftype1 == 0.}
    \item{ftype2}{Binary indicator variable used here to represent farm type in terms of the predominant susceptible species held, i.e. ftype1 == 2 if predominant species is sheep, otherwise ftype2 == 0.}
    \item{herdn}{Number of susceptible animals held on the farm at the start of the outbreak.}
    \item{status}{Status of the farm at the start of the period under observation: 1 == Susceptible, 2 == Exposed, 3 == Infectious, 4 == Recovered/Removed.}
  }
}

\references{
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\keyword{datasets}
