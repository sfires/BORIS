infer.checkInputs <- function(parsAux , keyInits , priors , scalingFactors, seed){
  
  if ((parsAux[1, "n"] < 1) | (parsAux[1, "n"] > 500)){ stop("Population size, n, is out of acceptable range of [1 , 500]!")}
  if ( sum(parsAux[1, "kernel_type"] == c("exponential", "cauchy" , "gaussian" , "power_law") ) == 0 )  { stop("Kernel type can only take one of 'exponential', 'cauchy' , 'gaussian' , 'power_law'! ")}
  if ( sum(parsAux[1, "coord_type"] == c("longlat","cartesian")) == 0 ){ stop("Coordinate system can only take 'longlat' for decimal degrees (i.e. latitudes and longitudes) or 'cartesian' for projected coordinate reference system!")}
  if ((parsAux[1, "t_max"] < 1 ) | !is.numeric(parsAux[1, "t_max"])){ stop("Upper limit of observation period, t_max, must be a positive number!")}
  if (!is.numeric(parsAux[1, "unassigned_time"]) | (parsAux[1, "unassigned_time"] < 1 )){ stop("The extreme value to indicate that an event does not happen, unassigned_time, must be a positive number!")}

  if (parsAux[1, "processes"] < 1 ){ stop("Number of parallel processes must be a positive integer!")}
  if (parsAux[1, "processes"] > 1 ){ stop("Only use processes > 1 for a SLURM cluster computer implementation. For multiple chains on other systems, use process = 1 and run several instances with different first values of the argument 'seed'.")}
  
  if ((parsAux[1, "n_seq"] < 1 ) | (parsAux[1, "n_seq"] >10 )){ stop("Initial number of sequences expected for a farm, n_seq, must be in between 1 and 10. This is redefined if more memory needs to be allocated for specific farms, so best set as low as expected!")}
  if ((parsAux[1, "n_base"] < 1 ) | (parsAux[1, "n_base"] > 10000 )){ stop("Number of bases (nucleotides) in length for a sequence, n_base,  is restricted between 1 and 10,000. For longer sequences, use only partial sequences of SNPs!")}
  if (parsAux[1, "n_iterations"] < 1 ){ stop("Number of iterations of the MCMC, n_iterations, must be a positive integer!")}
  if ((parsAux[1, "n_iterations"] >= 1) & (parsAux[1, "n_iterations"] <1e5) ){ cat("Warning: Are you sure you want such a low number MCMC iterations?\n")}
  if ((parsAux[1, "n_frequ"] < 1 ) | (parsAux[1, "n_frequ"] >  parsAux[1, "n_iterations"] )){ stop("Frequency of updating exposure times, n_frequ, must be in between 1 and n_iterations!")}
  if ((parsAux[1, "n_output_source"] < 1 ) | (parsAux[1, "n_output_source"] >  parsAux[1, "n_iterations"] )){ stop("Frequency of outputing updated sources for each known infected farm, n_output_source, must be in between 1 and n_iterations!")}
  if ((parsAux[1, "n_output_gm"] < 1 ) | (parsAux[1, "n_output_gm"] >  parsAux[1, "n_iterations"] )){ stop("Frequency of outputting an update of the grand master sequence (Gm), n_output_gm, must be in between 1 and n_iterations!")}
  if ((parsAux[1, "n_cout"] < 1 ) | (parsAux[1, "n_cout"] >  parsAux[1, "n_iterations"] )){ stop("Frequency of updating the console output, n_cout, must be in between 1 and n_iterations!")}
  if ( sum(parsAux[1, "opt_latgamma"] == c(0,1)) == 0 ){ stop("Implementation option, opt_latgamma, can take 1 to assume a Gamma distribution for the latent period or 0 to assume a Gaussian distribution!")}
  if ( sum(parsAux[1, "opt_k80"] == c(0,1) ) == 0 ){ stop("Implementation option, opt_k80, can take 1 for reformulated K80 DNA substitution model to match original 1980 paper or 0 for the original version in Lau et al. (2015, see the references of the package) based on a secondary reference!")}
 if ( sum(parsAux[1, "opt_betaij"] == c(0,1) ) == 0 ){ stop("Implementation option, opt_betaij, can take 1 for farm-level covariates incorporated into beta, i.e. betaij Lau modified model from Firestone et al. (2019, see the references of the package), 0 for the originally implemented model from Lau et al. (2015, see the references of the package.")}
  if ( sum(parsAux[1, "opt_ti_update"] == c(0,1) ) == 0 ){ stop("Implementation option opt_ti_update 1 to update timing of inferred onset of infectivity or 0 as in Lau original implementation based on simulated data, see Supporting Information in Lau et al. (2015, see the references of the package)!")}
  if ( sum(parsAux[1, "opt_mov"] == c(0,1,2) ) == 0 ){ stop("Implementation option opt_mov, can take 1 for contact/movement network to be incorprated into likelihood, or 0 for the originally implemented model from Lau et al. (2015, see the references of the package)")}  
  
  # ---------------------------------------------------------- #
  if ((keyInits[1, "alpha"] < 0 ) ){ stop("Initial value for the background (primary) transmission rate of infection, alpha, must be positive!")}
  if ((keyInits[1, "beta"] <= 0 ) | (keyInits[1, "beta"] > 10 )){ stop("Initial value for the secondary transmission rate, beta, must be in between 0 and 30!")}
  if ((keyInits[1, "lat_mu"] <= 0 ) | (keyInits[1, "lat_mu"] > 100 )){ stop("Initial value for the mean of the duration of the farm-level latent period, lat_mu,  must be in between 0 and 100!")}
  if ((keyInits[1, "lat_var"] <= 0 ) | (keyInits[1, "lat_var"] > 100 )){ stop("Initial value for the variance of the duration of the farm-level latent period, lat_var, must be in between 0 and 100!")}
  if ((keyInits[1, "c"] <= 0 ) | (keyInits[1, "c"] > 3650 )){ stop("Initial value for the scale parameter of the Weibull distribution representing the mean infectious period, c, must be in between 0 and 3650!")}
  if ((keyInits[1, "d"] <= 0 ) | (keyInits[1, "d"] > 100 )){ stop("Initial value for the shape parameter of the Weibull distribution representing the mean infectious period, d, must be in between 0 and 100!")}
  if ((keyInits[1, "k_1"] < 0 ) | (keyInits[1, "k_1"] > 10 )){ stop("Initial value for the spatial transmission kernel shape parameter, k_1, must be in between 0 and 10!")}
  if ((keyInits[1, "mu_1"] <= 0 ) | (keyInits[1, "mu_1"] > 1.00E-02 )){ stop("Initial value for the rate of transition mutations based on Kimura's 2 parameter nucleotide substitution model (Kimura, 1980), mu_1, must be in between 0 and 0.01!")}
  if ((keyInits[1, "mu_2"] <= 0 ) | (keyInits[1, "mu_2"] > 1.00E-02 )){ stop("Initial value for the rate of transversion mutations based on Kimura's 2 parameter nucleotide substitution model (Kimura, 1980), mu_2, must be in between  0 and 0.01!")}
  if ((keyInits[1, "p_ber"] <= 0 ) | (keyInits[1, "p_ber"] > 1 )){ stop("Initial value for the  probability that a nucleotide base of each of the primary (seeding) sequences has of differing from the base at the corresponding site in the sequence of the universal master sequence, p_ber, must be in between 0 and 1!")}
  if (keyInits[1, "phi_inf1"] <= 0 ) { stop("Initial value for the multiplicative effect of predominant species on premises-level infectivity compared to a reference of ftype0 farms, for ftype1 farms, phi_inf1, must be greatre than 0!")}
  if (keyInits[1, "phi_inf2"] <= 0 ){ stop("Initial value for the multiplicative effect of predominant species on premises-level infectivity compared to a reference of ftype0 farms, for ftype2 farms, phi_inf2, must be greatre than 0!")}
  if (keyInits[1, "rho_susc1"] <= 0 ){ stop("Initial value for the multiplicative effect of predominant species on premises-level susceptibility compared to a reference of ftype0 farms, for ftype1 farms, rho_susc1, must be greatre than 0!")}
  if (keyInits[1, "rho_susc2"] <= 0 ){ stop("Initial value for the multiplicative effect of predominant species on premises-level susceptibility compared to a reference of ftype0 farms, for ftype2 farms, rho_susc2, must be greatre than 0!")}
  if ((keyInits[1, "nu_inf"] < 0 ) | (keyInits[1, "nu_inf"] > 1 )){ stop("Initial value for the effect (power) of number of animals on premises-level infectivity for farms, nu_inf, must be in between 0 and 1!")}
  if ((keyInits[1, "tau_susc"] < 0 ) | (keyInits[1, "tau_susc"] > 1 )){ stop("Initial value for the effect (power) of number of animals on premises-level susceptibility for farms, tau_susc, must be in between 0 and 1!")}
  if ((keyInits[1, "beta_m"] < 0 )){ stop("Initial value for the transmission rate related to animal movements/contact-tracing, beta_m, cannot be less than or equal to 0!")}

  # ---------------------------------------------------------- #    
  if (priors[1, "t_range"] <= 0){ stop("The parameter t_range cannot be less than or equal to 0!")}
  if (priors[1, "t_back"] <= 0){ stop("Maximum assumed length for the latent period, t_back, cannot be less than or equal to 0!")}
   if (priors[1, "t_bound_hi"] < 1 ){ stop("The parameter t_bound_hi cannot be less than 1!")}
  if (priors[1, "rate_exp_prior"] <= 0){ stop("The rate of exposure, rate_exp_prior, cannot be less than or equal to 0!")}
  if ( sum(priors[1, "ind_n_base_part"] == c(0, 1) ) == 0 ){ stop("Logical, ind_n_base_part, representing whether the genomic sequence data is partial (=1) or complete (=0) must be either 0 or 1!")}
  if (priors[1, "n_base_part"] < 1 ){ stop("The partial sequence length, n_base_part, cannot be less than 1!")}
  if ((priors[1, "alpha_hi"] <= 0 ) | (priors[1, "alpha_hi"] > 10 )){ stop("The upper bound for the parameter alpha_hi must be in between (0, 10]!")}
  if ((priors[1, "beta_hi"] <= 0 ) | (priors[1, "beta_hi"] > 100 )){ stop("The upper bound for the parameter beta_hi must be in between (0, 100]!")}
  if ((priors[1, "mu_lat_hi"] <= 0 ) | (priors[1, "mu_lat_hi"] > 100 )){ stop("The upper bound for the parameter mu_lat_hi must be in between (0, 100]!")}
  if ((priors[1, "var_lat_lo"] <= 0 ) | (priors[1, "var_lat_lo"] > 10 )){ stop("The lower bound for parameter var_lat_lo must be in between (0, 10]!")}
  if ((priors[1, "var_lat_hi"] <= 0 ) | (priors[1, "var_lat_hi"] > 100 )){ stop("The upper bound for the parameter var_lat_hi must be in between (0, 100]!")}
  if ((priors[1, "c_hi"] <= 0 ) | (priors[1, "c_hi"] > 3650 )){ stop("The upper bound for the parameter c_hi must be in between (0, 3650]!")}
  if ((priors[1, "k_1_hi"] <= 0 ) | (priors[1, "k_1_hi"] > 1000 )){ stop("The upper bound for the parameter k_1_hi must be in between (0, 10]!")}
  if ((priors[1, "mu_1_hi"] <= 0 ) | (priors[1, "mu_1_hi"] > 1000 )){ stop("The upper bound for the parameter mu_1_hi must be in between (0, 10]!")}
  if ((priors[1, "mu_2_hi"] <= 0 ) | (priors[1, "mu_2_hi"] > 1000 )){ stop("The upper bound for the parameter mu_2_hi must be in between (0, 10]!")}
  if ((priors[1, "p_ber_hi"] <= 0 ) | (priors[1, "p_ber_hi"] > 1 )){ stop("The upper bound for the parameter p_ber_hi must be in between (0, 10]!")}
  if (priors[1, "phi_inf1_hi"] <= 0 ){ stop("The upper bound for the parameter phi_inf1_hi cannot be less than or equal to 0!")}
  if (priors[1, "phi_inf2_hi"] <= 0 ){ stop("The upper bound for the parameter phi_inf2_hi cannot be less than or equal to 0!")}
  if (priors[1, "rho_susc1_hi"] <= 0 ){ stop("The upper bound for the parameter rho_susc1_hi cannot be less than or equal to 0!")}
  if (priors[1, "rho_susc2_hi"] <= 0 ){ stop("The upper bound for the  parameter rho_susc2_hi cannot be less than or equal to 0! ")}
  if ((priors[1, "nu_inf_lo"] < 0 ) | (priors[1, "nu_inf_lo"] > 1 )){ stop("The lower bound for the parameter nu_inf_lo must be in between [0, 1]!")}
  if ((priors[1, "nu_inf_hi"] < priors[1, "nu_inf_lo"] ) | (priors[1, "nu_inf_hi"] > 1 )){ stop("The upper bound for the parameter nu_inf_hi must be in between(nu_inf_lo, 1]!")}  
  
  if ((priors[1, "tau_susc_lo"] < 0 ) | (priors[1, "tau_susc_lo"] > 1 )){ stop("The lower bound for the parameter tau_susc_lo must be in between [0, 1]!")}
  if ((priors[1, "tau_susc_hi"] < priors[1, "tau_susc_lo"] ) | (priors[1, "tau_susc_hi"] > 1 )){ stop("The upper bound for the parameter tau_susc_hi must be in between(tau_susc_lo, 1]!")}  

  if (priors[1, "beta_m_hi"] <= 0 ){ stop("The upper bound for the  parameter beta_m_hi cannot be less than or equal to 0! ")}
  
  if (priors[1, "trace_window"] <= 0 ){ stop("The upper bound for the  parameter trace_window cannot be less than or equal to 0! ")}
  # ---------------------------------------------------------- #    
  
  if (scalingFactors[1, "alpha_sf"] < 0){ stop("The scaling factor, alpha_sf, cannot be less than 0!")}
  if (scalingFactors[1, "beta_sf"] < 0){ stop("The scaling factor, beta_sf, cannot be less than 0!")}
  if (scalingFactors[1, "mu_lat_sf"] < 0){ stop("The scaling factor, mu_lat_sf, cannot be less than 0!")}
  if (scalingFactors[1, "var_lat_sf"] < 0){ stop("The scaling factor, var_lat_sf, cannot be less than 0!")}
  if (scalingFactors[1, "c_sf"] < 0){ stop("The scaling factor, c_sf, cannot be less than 0!")}
  if (scalingFactors[1, "k_1_sf"] < 0){ stop("The scaling factor, k_1_sf, cannot be less than 0!")}
  if (scalingFactors[1, "mu_1_sf"] < 0){ stop("The scaling factor, mu_1_sf, cannot be less than 0!")}
  if (scalingFactors[1, "mu_2_sf"] < 0){ stop("The scaling factor, mu_2_sf, cannot be less than 0!")}
  if (scalingFactors[1, "p_ber_sf"] < 0){ stop("The scaling factor, p_ber_sf, cannot be less than 0!")}
  if (scalingFactors[1, "phi_inf1_sf"] < 0){ stop("The scaling factor, phi_inf1_sf, cannot be less than 0!")}
  if (scalingFactors[1, "phi_inf2_sf"] < 0){ stop("The scaling factor, phi_inf2_sf, cannot be less than 0!")}
  if (scalingFactors[1, "rho_susc1_sf"] < 0){ stop("The scaling factor, rho_susc1_sf, cannot be less than 0!")}
  if (scalingFactors[1, "rho_susc2_sf"] < 0){ stop("The scaling factor, rho_susc2_sf, cannot be less than 0!")}
  if ((scalingFactors[1, "nu_inf_sf"] < 0 ) | (scalingFactors[1, "nu_inf_sf"] > 1 )){ stop("The scaling factor, nu_inf_sf, must be in [0, 1]!")}
  if ((scalingFactors[1, "tau_susc_sf"] < 0 ) | (scalingFactors["tau_susc_sf"] > 1 )){ stop("The scaling factor, tau_susc_sf, must be in [0, 1]!")}
  if (scalingFactors[1, "beta_m_sf"] < 0){ stop("The scaling factor, beta_m_sf, cannot be less than 0!")}  


# ---------------------------------------------------------- #    

if (seed[1] < 0){ stop("The seed must be greater than 0!")}  

}
