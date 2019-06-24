## ---- include = FALSE----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----sim_setup1----------------------------------------------------------
library(BORIS)

data(sim.epi.input)

head(sim.epi.input)


data(sim.moves.input)

head(sim.moves.input)

## ----sim_setup2----------------------------------------------------------
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

## ----sim_setup3----------------------------------------------------------
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
                       stringsAsFactors = F)

## ----sim-----------------------------------------------------------------
sim.out<-sim(epi.inputs = sim.epi.input, 
             moves.inputs = sim.moves.input, 
             parsKey = para.key,
             parsAux = pars.aux, 
             inputPath = "./inputs", 
             outputPath = "./outputs")

## ----sim_out1, eval=F----------------------------------------------------
#  sim.out$epi.sim[1:6,]

## ----sim_out2, echo=F----------------------------------------------------
tmp<-sim.out$epi.sim[1:6,]
tmp[,4:6]<-round(tmp[,4:6],4)
tmp

## ----sim_out3------------------------------------------------------------
head(sim.out$infected_source, 14)

## ----sim_out4------------------------------------------------------------
which(sim.out$infected_source == -99)

## ----sim_out5------------------------------------------------------------
inf <- which(sim.out$infected_source != -99)
length(inf)/length(sim.out$infected_source)

## ----sim_out6------------------------------------------------------------
sim.out$sampled_perct

## ----sim_out7------------------------------------------------------------
head(floor(sim.out$t_sample), 20)

## ----sim_out8------------------------------------------------------------
length(which(sim.out$t_sample[inf] != 9e6))/length(inf)

## ----sim_apeout----------------------------------------------------------
library(ape)

#read in the sequences from individual k=0
fn <- paste0(sim.out$outputPath, "subject_0_nt.txt")
seqs1 <- as.matrix(read.csv(fn, header=F))

#the sequences for this individul are stored in 19 rows
# nucleotides are coded as 1=A, 2=G, 3=T, 4=C
dim(seqs1)

#convert the sequences to their nucleotide base letters
seqs1[seqs1==1]<-"a"
seqs1[seqs1==2]<-"g"
seqs1[seqs1==3]<-"t"
seqs1[seqs1==4]<-"c"

#name the rows, which become the tip labels in the fasta file.
#for this we will import the timings associated with each sequence
fn <- paste0(sim.out$outputPath, "subject_0_t_nt.txt")
seqs1_t <- as.numeric(unlist(read.table(fn)))

rownames(seqs1) <- paste0("k0_", round(seqs1_t,3))

#write these out to a fasta file
library(ape)
write.dna(seqs1, file='seqs1.fasta', format = 'fasta', nbcol = -1)

#inspect snps in these data
seq.tab <- apply(seqs1, MARGIN = 2, table)
snp.v<-sapply(seq.tab, length)
snps<-as.numeric(which(snp.v>1))
seqs1[1:8,snps]

## ----inf_setup1----------------------------------------------------------
library(BORIS)

#use the epidemiological data from the previous simulation run
epi.data<-as.data.frame(sim.out$epi.sim)

#produce the required temporal fields. Here we are assuming that 
#onset of infectiousness typically occurs 24 hours before the 
#first clinical signs
epi.data$t_o <- floor(epi.data$t_i) + 1
epi.data$t_s <- round(sim.out$t_sample, 0)
epi.data$t_r <- round(epi.data$t_r, 0)

#wrangle the farm type variables into the required format
epi.data$ftype <- NA
epi.data$ftype[epi.data$ftype0 == 1] <- 0
epi.data$ftype[epi.data$ftype1 == 1] <- 1
epi.data$ftype[epi.data$ftype2 == 1] <- 2

#keep just the desired columns, re-ordered as required
epi.data<-epi.data[,c('k','coor_x','coor_y','t_o','t_s','t_r','ftype','herdn')]
head(epi.data)

#use the movement data from previously
moves.inputs<-sim.out$moves.inputs
head(moves.inputs)

## ----inf_setup2----------------------------------------------------------
pars.aux <- data.frame('n' = nrow(epi.data),
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
                       stringsAsFactors = F)

## ----inf_setup3----------------------------------------------------------
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

## ----inf_setup4----------------------------------------------------------
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

## ----inf_setup5----------------------------------------------------------
para.sf <- data.frame('alpha_sf' = 0.001,
                      'beta_sf' = 0.5,
                      'mu_lat_sf'	= 1.25, 
                      'var_lat_sf' = 1.75,
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

## ----inf_setup6----------------------------------------------------------
t_sample <- sim.out$t_sample

#which individuals are missing genomes and which were sampled
no.genome <- which(t_sample == pars.aux$unassigned_time)  
genome.sampled <- which(t_sample != pars.aux$unassigned_time)  

#the sampling times of those with genomes
round(t_sample[-no.genome], 3)

#setup a matrix to hold 1 sequence per individual or missing data
seq_mat<-matrix(nrow=nrow(epi.data), ncol=pars.aux$n_base)

#identify and store the data from sequences at the time of sampling 
#the next step takes approximately 30 seconds to run
for(i in 1:nrow(epi.data)){
  k <- i-1
  if(i %in% no.genome){
    #these are not used, given t.sample for these = the unassigned time
    #n denotes any base (in the IUPAC system)
    seq_mat[i,]<-rep("n", pars.aux$n_base)
  }else{
    #identify the closest corresponding sample to the time of 
    #sampling for each sampled individual
    fn<-paste0(sim.out$outputPath,"subject_",k,"_t_nt.txt")
    ts<-as.numeric(unlist(read.csv(fn, header=F)))
    tsamp <- which.min(abs(ts - t_sample[i]))
    #read in the corresponding nucleotide sequence
    fn<-paste0(sim.out$outputPath,"subject_",k,"_nt.txt")
    nts<-read.csv(fn, header = F, stringsAsFactors = F)
    #store the closest corresponding sample for this individual
    seq_mat[i,]<-as.numeric(nts[tsamp,])
  }
  if((i%%2) == 0) {cat(".",sep=""); flush.console()}  #
} 

## ----inf_setup7----------------------------------------------------------

#convert the sequence data to their nucleotide base letters
seq_mat[seq_mat==1]<-"a"
seq_mat[seq_mat==2]<-"g"
seq_mat[seq_mat==3]<-"t"
seq_mat[seq_mat==4]<-"c"

#write row.names to the sequence matrix object, including the 'k' 
#and time of sampling. These will be used as tip labels
row.names(seq_mat)<- paste0(epi.data$k, "_", epi.data$t_s)


if(!dir.exists("gen_inputs")) {
      dir.create("gen_inputs")
    } 


#write these out to a fasta file in the dnaPath directory
library(ape)
write.dna(seq_mat, file='./gen_inputs/seqs.fasta', format = 'fasta', nbcol = -1)


## ----inf_run1, eval=F----------------------------------------------------
#  pars.aux$n_iterations = 1000
#  
#  infer.out<-infer(covariates = epi.data,
#                   moves.inputs = moves.inputs,
#                   parsAux = pars.aux,
#                   keyInits = para.key.inits,
#                   priors = para.priors,
#                   scalingFactors = para.sf,
#                   seed = 1,
#                   accTable = sim.out$infected_source,
#                   t.sample = sim.out$t_sample,
#                   inputPath = "./inputs",
#                   outputPath = "./outputs",
#                   dnaPath = "./gen_inputs")

## ----inf_run2, eval=F----------------------------------------------------
#  task.id <- as.numeric(Sys.getenv('SLURM_ARRAY_TASK_ID'))
#  task.id
#  
#  .seed <- c(1, 102, 1003, 10004)[task.id]
#  .seed

## ----inf_post_chain1-----------------------------------------------------
library(BORIS)
data(paras1)
data(paras2)

burnin=0
thin = 1
end=1e04

x<-seq(burnin+1,end,thin)


par(mfrow=c(2,2))
plot(x, paras1$alpha[x], type='l', col='#ff000060', 
     ylim=c(0,0.01), 
     xlab='iteration', ylab=expression(alpha), cex.lab=1.4)
lines(x, paras2$alpha[x], col='#0000ff60')

plot(x, paras1$beta[x], type='l', col='#ff000060',
     xlab='iteration', ylab=expression(beta),
     cex.lab=1.4)
lines(x, paras2$beta[x], col='#0000ff60')

plot(x, paras1$mu_1[x], type='l', col='#ff000060',
     xlab='iteration', ylab=expression(mu[1]),
     cex.lab=1.4)
lines(x, paras2$mu_1[x], col='#0000ff60')

plot(x, paras1$p_ber[x], type='l', col='#ff000060',
     xlab='iteration', ylab=expression(p[ber]),
     cex.lab=1.4)
lines(x, paras2$p_ber[x], col='#0000ff60')


## ----inf_post_chain2-----------------------------------------------------
burnin=5000
thin = 1
end=1e04
x<-seq(burnin+1,end,thin)

paras<-rbind(paras1[x,], paras2[x,])

#posterior estimates for alpha
round(quantile(paras$alpha, probs=c(0.5, 0.025, 0.975)), 4)

#posterior estimates for beta
round(quantile(paras$beta, probs=c(0.5, 0.025, 0.975)), 3)

#posterior estimates for c
round(quantile(paras$c, probs=c(0.5, 0.025, 0.975)), 2)

#posterior estimates for d
round(quantile(paras$d, probs=c(0.5, 0.025, 0.975)), 3)

#posterior estimates for mu_1
quantile(paras$mu_1, probs=c(0.5, 0.025, 0.975))

#posterior estimates for mu_2
quantile(paras$mu_2, probs=c(0.5, 0.025, 0.975))

## ----inf_post_chain3-----------------------------------------------------
plot(density(paras1$alpha[x]), col='red', cex.main=2, 
  main=expression(paste(plain("Density of "),alpha,plain(", by run"))))
lines(density(paras2$alpha[x]), col='blue')

## ----inf_post_sources1---------------------------------------------------
data(inf1)
head(inf1[,1:10])
tail(inf1[,1:10])

data(inf2)
head(inf2[,1:10])
tail(inf2[,1:10])

## ----inf_post_sources2---------------------------------------------------
burnin=5000
thin = 1
end=1e04
x<-seq(burnin+1,end,thin)

#create lists that store for each individual:
#- a frequence table of iterations featuring each infected source (inf.l)
#- the source with the highest modal posterior support (inf.mode.l)
#- the posterior support for each infected source (inf.support.l)
inf.l<-list()
inf.mode.l<-numeric()
inf.support.l<-numeric()
for(i in 1:nrow(epi.data)){  
  inf.l[[i]]<-table(c(inf1[x,i], inf2[x,i]))
  inf.mode.l[i]<-as.numeric(attr(which.max(inf.l[[i]]), 'names'))
  inf.support.l[i]<-
    as.numeric(inf.l[[i]][which.max(inf.l[[i]])]/sum(inf.l[[i]]))
}

#inspect these outputs for individual k=6 
inf.l[[2]]
inf.mode.l[2]
inf.support.l[2]

## ----inf_post_sources3---------------------------------------------------
el<-data.frame(to=1:nrow(epi.data), from=inf.mode.l)

head(el)

## ----inf_post_sources4---------------------------------------------------
#from the simulation
accTable <- sim.out$infected_source

#which are correctly inferred
acc <- ifelse(accTable == el$from, 1, 0)
acc

#number correct
sum(acc)

round(sum(acc)/length(el$from),2)

## ----inf_post_sources5---------------------------------------------------
nsup<-length(which(inf.support.l>0.5))
nsup

#proportion of individuals with an inferred source with consensus support
round(nsup/nrow(epi.data),2)

#accuracy for the inferred sources of these individuals only
round(sum(acc[inf.support.l>0.5])/nsup, 2)

## ----inf_post_timings1---------------------------------------------------
data(te1)
round(head(te1[,1:10]), 3)
round(tail(te1[,1:10]), 3)

data(te2)
round(head(te2[,1:10]), 3)
round(tail(te2[,1:10]), 3)

## ----inf_post_timings2---------------------------------------------------
burnin=5000
thin = 1
end=1e04
x<-seq(burnin+1,end,thin)

te.l<-list()
te.q.l<-list()
for(i in 1:nrow(epi.data)){  
  te.l[[i]]<-c(te1[x,i], te2[x,i])
  te.q.l[[i]]<-quantile(te.l[[i]], probs=c(0.5, 0.025, 0.975))
}

#inspecting some of the inferred exposure timings
round(te.q.l[[1]], 2)
round(te.q.l[[2]], 2)
round(te.q.l[[3]], 2)

## ----inf_post_timings3---------------------------------------------------
data(ti1)
round(head(ti1[,1:10]), 3)
round(tail(ti1[,1:10]), 3)

data(ti2)
round(head(ti2[,1:10]), 3)
round(tail(ti2[,1:10]), 3)

## ----inf_post_timings4---------------------------------------------------
burnin=5000
thin = 1
end=1e04
x<-seq(burnin+1,end,thin)

ti.l<-list()
ti.q.l<-list()
for(i in 1:nrow(epi.data)){  
  ti.l[[i]]<-c(ti1[x,i], ti2[x,i])
  ti.q.l[[i]]<-quantile(ti.l[[i]], probs=c(0.5, 0.025, 0.975))
}

#inspecting some of the inferred exposure timings
round(ti.q.l[[1]], 2)
round(ti.q.l[[2]], 2)
round(ti.q.l[[3]], 2)

## ----inf_post_seqs_1-----------------------------------------------------
data(nt.t1)

head(nt.t1)

nt.t.l<-apply(nt.t1, MARGIN=1, 
              FUN = function(x) as.numeric(unlist(strsplit(x, ","))))

# the timings corresponding to the sequences for the first three individuals
# in the first iteration
nt.t.l[[1]]
nt.t.l[[2]]
nt.t.l[[3]]

# the timings corresponding to the sequences for the first
# individual in the first three iterations
nt.t.l[[pars.aux$n*0+1]]
nt.t.l[[pars.aux$n*1+1]]
nt.t.l[[pars.aux$n*2+1]]

#this indiviual was sampled at
t.sample<-round(sim.out$t_sample, 0)
t.sample[1]

## ----inf_post_seqs_2-----------------------------------------------------
nt.size<-sapply(nt.t.l, length)

nt.size[[pars.aux$n*0+1]]
nt.size[[pars.aux$n*1+1]]
nt.size[[pars.aux$n*2+1]]

## ----inf_post_seqs_3-----------------------------------------------------
#total number of sequences recorded
sum(nt.size)

#matrix to hold number of sequences per individual (column) per iteration (row)
nt.size.mat<-matrix(nt.size, byrow = T, ncol=pars.aux$n)
nt.size.mat[1:2,1:10]

#total number of sequences stored per iteration
nt.size.mat.sizes <- apply(nt.size.mat, MARGIN = 1, FUN = sum)
nt.size.mat.sizes

## ----inf_post_seqs_4-----------------------------------------------------
data(nt.seq1)

#1 sequence per row, corresponding to the timings in nt.t1
dim(nt.seq1)

## ----inf_post_seqs_5-----------------------------------------------------
nt.size[[1]]
nt.seq1[1:5,1:10]

## ----inf_post_seqs_6-----------------------------------------------------
nt.size[[1]]
nt.size[[2]]
nt.seq1[6:8,1:10]

## ----inf_post_seqs_7-----------------------------------------------------
nt.size[[pars.aux$n*1+2]]

## ----inf_post_seqs_8-----------------------------------------------------
#count sequences in the first iteration
.row.nos <- nt.size.mat.sizes[1]
#add sequences for the first individual in the 2nd iteration
.row.nos <- .row.nos + nt.size[[pars.aux$n*1+1]]
#count sequences for the second individual in the 2nd iteration
.row.nos <- .row.nos + 1:(nt.size[[pars.aux$n*1+2]])
.row.nos

nt.seq1[.row.nos,1:10]

## ----inf_post_seqs_9-----------------------------------------------------
nt.size[[pars.aux$n*2+3]]

## ----inf_post_seqs_10----------------------------------------------------
#count sequences in the earlier iterations
.row.nos <- sum(nt.size.mat.sizes[1:2])
#add sequences for the first individual in the 3nd iteration
.row.nos <- .row.nos + nt.size[[pars.aux$n*2+1]]
#count sequences for the second individual in the 3nd iteration
.row.nos <- .row.nos + 1:(nt.size[[pars.aux$n*2+2]])
.row.nos

nt.seq1[.row.nos,1:10]

