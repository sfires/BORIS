seq.lookup<-function(k=0, it=1, seq.ts, accTable){
  ##looking up timings
  #no. of individuals
  n <- length(accTable)
  #no. with genomes
  gs <- which(accTable != -99)
  n.g <- length(gs)
  
  ##looking up sequence locations per timing
  #manipulate the timings data to make it manageable
  nt.t.l<-strsplit(trimws(seq.ts),",")
  #timings for k in iteration 'it'
  ts<-as.numeric(nt.t.l[[n.g*(it-1)+(k+1)]])
  
  # how many sequences per individual per iteration
  nt.size<-sapply(nt.t.l, length)
  #matrix to hold number of sequences per individual (column) per iteration (row)
  nt.size.mat<-matrix(nt.size, byrow = T, ncol=n.g)
  colnames(nt.size.mat)<- gs-1
  #nt.size.mat[1:2,1:10]
  #total number of sequences stored per iteration
  nt.size.mat.sizes <- apply(nt.size.mat, MARGIN=1, FUN=sum)
  if(it==1){.start<-0}
  if(it>1) {.start<-sum(nt.size.mat.sizes[1:(it-1)])}
  if(k==0) {.start<- .start + 1}
  if(k>=1) {.start<-.start + sum(nt.size.mat[it,1:k])}
  .stop<-.start + nt.size[n.g*(it-1)+(k+1)] - 1
  #sequence indices for k in iteration 'it'
  seq.ts <- .start:.stop
  #
  return(list(ts = ts, seq.ts = seq.ts))
}