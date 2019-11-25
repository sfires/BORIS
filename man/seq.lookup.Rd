\name{seq.lookup}
\alias{seq.lookup}
\title{Function to lookup inferred sequence data post run}

\description{
\code{seq.lookup} after a run of The \code{infer} function, this function looks up the timings of the sequences for a specified individual in a specified  iteration, and the corresponding row numbers of the sequence data, for subsequent lookup.
}

\usage{
  seq.lookup(k=0,
             it=1,
             seq.ts=NULL,
             accTable=NULL)
}

\arguments{
  \item{k}{The index of the individual for which sequences are desired.}
  \item{it}{The MCMC iteration number.}
  \item{seq.ts}{A \code{data.frame} of the sequence timing data output from the function \code{infer} and read in from the file \code{seqs_t_current.csv}. See example below.}
  \item{accTable}{A \code{data.frame} including known the sources of infection for each individual for simulated outbreaks, or \code{9999} for each unknown source. As specified in the function \code{infer}}
}

\details{
After a run of the function \code{infer}, sequence data are stored in a file \code{seqs_current.csv} which has a complicated structure because individuals typically have different numbers of sequences stored per iteration.

A sequence is written out every nth iteration, as set with the parameter \code{pars.aux$n_output_gm}.  In these iterations, whenever an individual is infected, sampled or when it infects another individual, a sequence is written. So each individual can have a different number of sequences recorded from other individuals, even from itself in different iterations.

Each \code{pars.aux$n_output_gm} iteration, the timing of recorded sequences is captured in the output file \code{seqs_t_current_csv} with 1 row per individual, comma-separated.

}

\value{
This function returns a list of sequence timings for the individual with index k in the nth iteration, and corresponding rows of the sequence data matrix, for subsequent lookup. See the example below.
\item{ts}{timings of sequences recorded for individual with index k in iteration it}
\item{seq.ts}{row numbers in the sequence data of sequences recorded for individual with index k in iteration it, that correspond to the timings in \code{ts}}
}

\references{

Firestone, S.M., Hayama, Y., Bradhurst, R.A., Yamamoto, T., Tsutsui, T., Stevenson, M.A., 2019. Reconstructing outbreaks: a methods comparison of transmission network models. Sci. Rep.

Kimura, M., 1980: A simple method for estimating evolutionary rates of base substitutions through comparative studies of nucleotide sequences. J Mol Evol, 16, 111-120.

Lau, M.S., Marion, G., Streftaris, G., Gibson, G., 2015. A systematic Bayesian integration of epidemiological and genetic data. PLoS Comput. Biol. 11, e1004633.
}

\author{
Simon Firestone
Maintainer: Simon Firestone <simon.firestone@unimelb.edu.au>
}

\examples{

\dontrun{
# read in the timings data from the first chain
nt.t<-readLines('seqs_t_current.csv')

#read in the sequence data
#this is a large file and takes several seconds to load
nt.seq<-read.csv('seqs_current.csv', header=F)

#check the dimensions of the sequence data
dim(nt.seq)

#convert the sequences to their nucleotide base letters
nt.seq[nt.seq==1]<-"a"
nt.seq[nt.seq==2]<-"g"
nt.seq[nt.seq==3]<-"t"
nt.seq[nt.seq==4]<-"c"

#use seq.lookup function to show timings of sequences recorded for individual with index `k` in iteration `it`
seq.lookup(k=1, it=2,seq.ts=nt.t, accTable=accTable)$ts

#show the first 10 nucleotides of the sequence data 
#for the corresponding sequences for individual k
#in iteration it
nt.seq1[seq.lookup(k=1, it=2,seq.ts=nt.t1,
                   accTable=accTable)$seq.ts,1:10]

}

# See package vignette for a complete example.

}