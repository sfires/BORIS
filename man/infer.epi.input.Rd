\name{infer.epi.input}

\docType{data}

\alias{infer.epi.input}

\title{Epidemiological (covariate) input data on farms in the population affected by the outbreak to be reconstructed.}

\description{
This data set provides an example of the inputs on the population of farms housing individual animals at risk for which an outbreak will be reconstructed and parameters inferred, along with corresponding genomic data.
}

\usage{data(infer.epi.input)}

\format{
  A data frame with 100 observations (1 per farm) on the following 11 variables:
  \describe{
    \item{k}{A unique identifier for each farm. Starts at zero given C++ indexing system.}
    \item{coor_x}{x coordinate of the farm's centroid,  e.g., Longitude in decimal degrees using WGS84 datum.}
    \item{coor_y}{y coordinate of the farm's centroid,  e.g., Latitude in decimal degrees using WGS84 datum.}
    \item{t_o}{Day of onset of first clinical signs observed in this individual (or in animals on this farm), since some arbitrary origin (Day 0). Can be left as the unassigned value (default \code{9e06}), for  uninfected farms or farms where clinical onset was not observed.}
    \item{t_s}{Day of sampling that led to diagnosis.}
    \item{t_r}{Day removed, recovered or culled, if known or the unassigned value (default \code{9e06}) if unknown.}
    \item{ftype}{A 3-level categorical variable used to represent farm type, here, in terms of the predominant susceptible species held, i.e. ftype = 0 if predominant species is cattle,  ftype = 1 if predominant species is pigs, ftype = 2 if predominant species is sheep/other. This variable is later converted into an indicator variable.}
    \item{herdn}{Number of susceptible animals held on the farm at the start of the outbreak.}
  }
}

\references{
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\keyword{datasets}
