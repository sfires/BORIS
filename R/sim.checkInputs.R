sim.checkInputs <- function(parsKey , parsAux){

  if ((parsKey[1, "alpha"] < 0 ) ){ stop("The background (primary) transmission rate of infection, alpha, must be positive!")}
  if ((parsKey[1, "beta"] <= 0 ) | (parsKey[1, "beta"] > 30 )){ stop("The secondary transmission rate, beta, must be in between 0 and 30!")}
  if ((parsKey[1, "mu_1"] <= 0 ) | (parsKey[1, "mu_1"] > 1.00E-02 )){ stop("The rate of transition mutations based on Kimura's 2 parameter nucleotide substitution model (Kimura, 1980), mu_1, must be in between 0 and 0.01!")}
  if ((parsKey[1, "mu_2"] <= 0 ) | (parsKey[1, "mu_2"] > 1.00E-02 )){ stop("The rate of transversion mutations based on Kimura's 2 parameter nucleotide substitution model (Kimura, 1980), mu_2, must be in between  0 and 0.01!")}
  if ((parsKey[1, "a"] <= 0 ) | (parsKey[1, "a"] > 1000 )){ stop("The shape parameter of the Gamma distribution representing the latent period, a, must be in between 0 and 1000!")}
  if ((parsKey[1, "b"] <= 0 ) | (parsKey[1, "b"] > 1000 )){ stop("The scale parameter of the Gamma distribution representing the latent period, b, must be in between 0 and 1000!")}
  if ((parsKey[1, "c"] <= 0 ) | (parsKey[1, "c"] > 1000 )){ stop("The scale parameter of Weibull distribution representing the mean infectious period, c, must be in between 0 and 1000!")}
  if ((parsKey[1, "d"] <= 0 ) | (parsKey[1, "d"] > 1000 )){ stop("The shape parameter of Weibull distribution representing the mean infectious period, d, must be in between 0 and 1000!")}
  if ((parsKey[1, "k_1"] <= 0 ) | (parsKey[1, "k_1"] > 10 )){ stop("The spatial transmission kernel shape parameter, k_1, must be in between 0 and 10!")}
  if ((parsKey[1, "p_ber"] <= 0 ) | (parsKey[1, "p_ber"] > 1 )){ stop("The  probability that a nucleotide base of each of the primary (seeding) sequences has of differing from the base at the corresponding site in the sequence of the universal master sequence, p_ber, must be in between 0 and 1!")}
  if (parsKey[1, "phi_inf1"] <= 0 ) { stop("The multiplicative effect of predominant species on premises-level infectivity compared to a reference of ftype0 farms, for ftype1 farms, phi_inf1, must be greater than 0!")}
  if (parsKey[1, "phi_inf2"] <= 0 ) { stop("The multiplicative effect of predominant species on premises-level infectivity compared to a reference of ftype0 farms, for ftype2 farms, phi_inf2, must be greater than 0!")}  
  if (parsKey[1, "rho_susc1"] <= 0 ) { stop("The multiplicative effect of predominant species on premises-level susceptibility compared to a reference of ftype0 farms, for ftype1 farms, rho_susc1, must be greater than 0!")}
  if (parsKey[1, "rho_susc2"] <= 0 ) { stop("The multiplicative effect of predominant species on premises-level susceptibility compared to a reference of ftype0 farms, for ftype2 farms, rho_susc2, must be greater than 0!")} 
  if ((parsKey[1, "nu_inf"] < 0 ) | (parsKey[1, "nu_inf"] > 1 )){ stop("The effect (power) of number of animals on premises-level infectivity for farms - nu_inf must be in between 0 and 1!")}
  if ((parsKey[1, "tau_susc"] < 0 ) | (parsKey[1, "tau_susc"] > 1 )){ stop("The effect (power) of number of animals on premises-level susceptibility for farms - tau_susc must be in between 0 and 1!")}  
  if ((parsKey[1, "beta_m"] < 0 ) | (parsKey[1, "beta_m"] > 30 )){ stop("The the secondary transmission rate by contact-related transmission/animal movement, beta_m, must be in between 0 and 30!")}

# ---------------------------------------------------------- #
  if ((parsAux[1, "n"] < 1) | (parsAux[1, "n"] > 500)){ stop("Population size, n, is out of acceptable range of [1 , 500]!")}
  if (!is.numeric(parsAux[1, "seed"]) | (parsAux[1, "seed"] < 1 )){ stop("The seed must be a positive integer!")}
  if ((parsAux[1, "n_base"] < 1 ) | (parsAux[1, "n_base"] > 10000 )){ stop("Number of bases (nucleotides) in length for a sequence, n_base,  is restricted between 1 and 10,000. For longer sequences, use only partial sequences of SNPs!")}
  if ((parsAux[1, "n_seq"] < 1 ) | (parsAux[1, "n_seq"] >10 )){ stop("Initial number of sequences expected for a farm, n_seq, must be in between 1 and 10. This is redefined if more memory needs to be allocated for specific farms, so best set as low as expected!")}
  if ((parsAux[1, "t_max"] < 1 ) | !is.numeric(parsAux[1, "t_max"])){ stop("Upper limit of observation period, t_max, must be a positive number!")}
  if (!is.numeric(parsAux[1, "unassigned_time"]) | (parsAux[1, "unassigned_time"] < 1 )){ stop("The extreme value to indicate that an event does not happen, unassigned_time, must be a positive number!")}
  if (!is.numeric(parsAux[1, "sample_range"]) | (parsAux[1, "sample_range"] < 1 )){ stop("The maximum possible delay between infection and sampling, sample_range, must be a positive number!")}
  if (!(parsAux[1, "partial_seq_out"] == 0 | parsAux[1, "partial_seq_out"] == 1)){ stop("Logical representing whether the genomic sequence data is partial (=1) or complete (=0) must be either 0 or 1!")}
  if (!is.numeric(parsAux[1, "n_base_part"]) | (parsAux[1, "n_base_part"] < 1) ){ stop("The partial sequence length, n_base_part, cannot be less than 1!")}
  if (!is.numeric(parsAux[1, "n_index"]) | (parsAux[1, "n_index"] < 1) | (parsAux[1, "n_index"] > .5*parsAux[1, "n"]) ){ stop("The number of indexes, n_index, cannot be less than 1 or greater than 50% of n!")}
  if ( sum(parsAux[1, "coord_type"] == c("longlat","cartesian")) == 0 ){ stop("Coordinate system can only take 'longlat' for decimal degrees (i.e. latitudes and longitudes) or 'cartesian' for projected coordinate reference systems!")}
  if ( sum(parsAux[1, "kernel_type"] == c("exponential", "cauchy" , "gaussian" , "power_law") ) == 0 )  { stop("Kernel type can only take one of 'exponential', 'cauchy' , 'gaussian' , 'power_law'!")}
  if ( sum(parsAux[1, "latent_type"] == c("gamma") ) == 0 )  { stop("Distribution type used for simulating the latent period presently only allows 'gamma'!")}
  if (!(parsAux[1, "opt_k80"] == 0 | parsAux[1, "opt_k80"] == 1)){ stop("Implementation option opt_k80 must be either 0 or 1!")}
  if (!(parsAux[1, "opt_betaij"] == 0 | parsAux[1, "opt_betaij"] == 1)){ stop("Implementation option opt_betaij must be either 0 or 1!")}
  if (!(parsAux[1, "opt_movt"] == 0 | parsAux[1, "opt_movt"] == 1)){ stop("Implementation option opt_movt must be either 0 or 1!")}
  if (!is.numeric(parsAux[1, "n_mov"]) | (parsAux[1, "n_mov"] < 1 )){ stop("The maximum possible number of animal movements or contacts, n_mov, must be a positive number, presently limited to 10,000!")}  

}
