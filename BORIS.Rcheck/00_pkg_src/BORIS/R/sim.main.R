sim.main<-function(){
  cpp_out<-sim_cpp()
  cat("\nOutbreak simulated successfully.\n ")
  return(list(epi.sim = cpp_out$epi.sim,
              infected_source = cpp_out$infected_source,
              t_sample = cpp_out$t_sample,
              sampled_perct = cpp_out$sampled_perct,
              cons_seq = cpp_out$cons_seq))
}
