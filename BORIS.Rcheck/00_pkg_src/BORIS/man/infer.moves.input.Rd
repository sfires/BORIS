\name{infer.moves.input}

\docType{data}

\alias{infer.moves.input}

\title{Movement (contract-tracing) input data between farms in the population affected by the outbreak to be reconstructed.}

\description{
This data set provides an example of inputs on the movements of animals between farms housing individual animals at risk in which an outbreak will be reconstructed and parameters inferred, along with corresponding genomic data.
}

\usage{data(infer.moves.input)}

\format{
  A data frame with 100 observations (1 per movement) on the following 3  variables:
  \describe{
    \item{from_k}{the unique identifier of the source farm for the animals moved. Starts at zero given C++ indexing system.}
    \item{to_k}{the unique identifier of the destination farm for the animals moved. Starts at zero given C++ indexing system.}
    \item{t_m}{day that the movement occurred.}
  }
}

\references{
Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.
  
Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\keyword{datasets}
