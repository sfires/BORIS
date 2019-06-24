infer.main<-function(){
  cpp_out<-infer_cpp()
  cat("\nAll requested chains have been generated successfully.\n ")
  return(list(seed = cpp_out$seed,
              t_sample = cpp_out$t_sample))
}