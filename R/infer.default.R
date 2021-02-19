#' @export
#' @useDynLib BORIS, .registration = TRUE
#' @importFrom ape read.dna
#' @importFrom utils write.table
infer.default <- function(covariates = NULL,
                          moves.inputs = NULL,
                          parsAux = NULL,
                          keyInits = NULL,
                          priors = NULL,
                          scalingFactors = NULL,
                          seed = NULL,
                          accTable = NULL,
                          t.sample = NULL,
                          inputPath = NULL,
                          outputPath = NULL,
                          dnaPath = NULL,
                          dnaReference = NULL,
                          debug = NULL){
  

  
  options(warn=-1)
  if ( .Platform$OS.type != "windows" ) {
    slash <- "/"
  } else {
    slash <- "\\"
  }
  if (is.null(covariates)){
    stop("Please supply the covariate data by the argument 'covariates'")
  }
  
  if (is.null(moves.inputs)){
    stop("Please supply the movement (contract-tracing) input data by the argument 'moves.inputs'. If 'opt_mov' is set to 0 these data will be ignored.")
  }

  if (is.null(parsAux)){
    stop("Please supply the auxiliary parameters by the argument 'parsAux'")
  }
  if (is.null(keyInits)){
    stop("Please supply the key initial values by the argument 'keyInits'")
  }  
  if (is.null(priors)){
    stop("Please supply the priors by the argument 'priors'")
  }
  if (is.null(scalingFactors)){
    stop("Please supply the scaling factors by the argument 'scalingFactors'")
  }
  
  if (is.null(seed)){
    stop("Please supply a seed with the argument 'seed'")
  }  
  if (is.null(accTable)){
    stop("Please supply the accuracy table by the argument 'accTable'")
  }
  
  if (is.null(t.sample)){
    stop("Please supply sampling times with the argument 't.sample'")
  }
  
  if (is.null(dnaPath)){
    stop("Please supply the file name/path for DNA data by the argument 'dnaPath'")
  }
  

  parsAux[1, "n_iterations"] = round(parsAux[1, "n_iterations"])
  
  
  infer.checkInputs(parsAux = parsAux,
                    keyInits = keyInits,
                    priors = priors,
                    scalingFactors = scalingFactors,
                    seed = seed)
  
  if (is.null(inputPath)){
    if(!dir.exists('inputs')) {dir.create('inputs')} 
    inputPath <- paste0(getwd(), slash , "inputs" , slash)
  } else {
    if (length(grep('inputs' , substr(inputPath,nchar(inputPath)-7,nchar(inputPath)))) == 0 ){
      if (substr(inputPath, nchar(inputPath), nchar(inputPath) ) != slash ){
        inputPath <- paste0(inputPath, slash, "inputs" , slash)  
      } else {
        inputPath <- paste0(inputPath, "inputs" , slash)  
      }
    } else {
      if (substr(inputPath, nchar(inputPath), nchar(inputPath) ) != slash ){
        inputPath <- paste0(inputPath, slash)
      }
    }
    if (length(dir(inputPath)) != 0){
      file.remove(paste0(inputPath, dir(inputPath)))
    }
    if(!dir.exists(inputPath)) {
      dir.create(inputPath)
    } 
  }
  cat("Input path:" , inputPath, "\n")
  
  if (is.null(outputPath)){
    if(!dir.exists('outputs')) {dir.create('outputs')} 
    outputPath <- paste0(getwd(), "/outputs")
  } else {
    if (length(grep('outputs' , substr(outputPath,nchar(outputPath)-8,nchar(outputPath)))) == 0 ){
      if (substr(outputPath, nchar(outputPath), nchar(outputPath) ) != slash ){
        outputPath <- paste0(outputPath, slash , "outputs" , slash) 
      } else {
        outputPath <- paste0(outputPath, "outputs" , slash)  
      }
    } else {
      if (substr(outputPath, nchar(outputPath), nchar(outputPath) ) != slash ){
        outputPath <- paste0(outputPath, slash)
      }
    }
    # if (file.exists(paste0(outputPath , "parameters_current.log"))){
    #   file.remove(paste0(outputPath , "parameters_current.log"))
    # }
    if (length(dir(outputPath)) != 0){
      file.remove(paste0(outputPath, dir(outputPath)))
    }
    if(!dir.exists(outputPath)) {
      dir.create(outputPath)
    } 
  }
  cat("Output path:" , outputPath, "\n")
  

##epi.data
  epi.data <- covariates
  #head(epi.data)
  
  #epi.data$t_i <- epi.data$t_o - 1
  #epi.data$t_e <- epi.data$t_i - 2
  
  epi.data$ftype0 <- 0
  epi.data$ftype0[epi.data$ftype == 0] <- 1
  epi.data$ftype1 <- 0
  epi.data$ftype1[epi.data$ftype == 1] <- 1
  epi.data$ftype2 <- 0
  epi.data$ftype2[epi.data$ftype == 2] <- 1

  #initial source
  # for (i in 1:nrow(epi.data)){
  #   #define pool of possible sources
  #   pool_source<- epi.data$k [which(epi.data$t_i<=epi.data$t_e[i] & epi.data$t_r>epi.data$t_e[i])]  
  #   # randomly assign an infection source
  #   if (length(pool_source)>1) epi.data$initial_source[i] <- sample(pool_source,1) 
  #   if (length(pool_source)==1) epi.data$initial_source[i] <- pool_source[1] 
  #   # background
  #   if (length(pool_source)<1) epi.data$initial_source[i] <- 9999 
  # }
  # # uninfected get -99
  # id <- which(epi.data$t_e == parsAux$unassigned_time)
  # epi.data$initial_source[id]<- -99
  epi.data$initial_source[is.na(epi.data$initial_source)]<-9999
    
    
  #reorder columns
  epi.data <- epi.data[,c("k", "coor_x", "coor_y", 
                          "t_e", "t_i", "t_r",
                          "ftype0", "ftype1", "ftype2", 
                          "herdn", "initial_source")]
  
  write.table(epi.data, paste0(inputPath, "/epi.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F,col.names = TRUE)

  write.table(epi.data[,c("coor_x", "coor_y")], paste0(inputPath, "/coordinate.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F,col.names = FALSE)
    
## seed   
  write.table(seed, paste(inputPath, "/seeds.csv", sep=''), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = F)
  
  #read sequence data
    if (length(grep('inputs' , substr(dnaPath,nchar(inputPath)-7,nchar(dnaPath)))) == 0 ){
      if (substr(dnaPath, nchar(dnaPath), nchar(dnaPath) ) != slash ){
        dnaPath <- paste0(dnaPath, slash, "gen_inputs" , slash)  
      } else {
        dnaPath <- paste0(dnaPath, "gen_inputs" , slash)  
      }
    } else {
      if (substr(dnaPath, nchar(dnaPath), nchar(dnaPath) ) != slash ){
        dnaPath <- paste0(dnaPath, slash)
      }
    }
    if(!dir.exists(dnaPath)) {
      stop("Check dnaPath and that a single .fasta file is present in dnaPath.")
    } 
  if(!dir.exists(dnaPath)) {
    stop("Check dnaPath and that a single .fasta file is present in dnaPath.")
  } 
  cat("dnaInput path:" , dnaPath, "\n")
  
  fn<-paste0(dnaPath, dir(dnaPath))
  seqs.sims.phylo <- ape::read.dna(fn, format='fasta', as.character = T)
  seq.l <- dim(seqs.sims.phylo)[2]

    
  ### atab
  write.table(accTable, paste(inputPath, "/atab_from.csv", sep=''), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = F)
  
  
  ##parameters_other.txt
  if(is.null(debug)){parsAux$debug = 0}
  if(!is.null(debug)){
    if(debug == 1){parsAux$debug = 1}else{parsAux$debug = 0}
  }
  
  write.table(parsAux, file = paste0(inputPath, "/parameters_other.csv", sep=''), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = names(parsAux) )
  

  ##index.txt
  index <- which(epi.data$initial_source == 9999) - 1
  index <- c('k', index)
  index.out<-matrix(index, nrow=length(index))
  write.table(index.out[1:2,], paste0(inputPath, "/index.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = F)
  
  ##t_sample.txt
  t_sample<-matrix(t.sample, nrow=parsAux$n)
  write.table(t_sample, paste0(inputPath, "/t_sample.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = F)
  
  ##seq.csv
  #turn sequence data into rows before further manipulation
  seq_out<-matrix(nrow=parsAux$n, ncol=seq.l)
  for(i in 1:parsAux$n){  seq_out[i,] <- seqs.sims.phylo[i,] }
  # convert as follows
  #1	A
  #2	G
  #3	T
  #4	C
  seq_out[seq_out == 'a'] <- 1
  seq_out[seq_out == 'g'] <- 2
  seq_out[seq_out == 't'] <- 3
  seq_out[seq_out == 'c'] <- 4
  seq_out[seq_out == 'n'] <- NA
  
  if(is.null(dnaReference)){
    
    write.table(seq_out, paste0(inputPath, "/seq.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = F)
    
    
    ##con_seq_estm.txt
    #estimate as most frequent base at each site
    most_freq <- function(x){as.numeric(names(sort(table(x),decreasing=TRUE)[1]))}
    con_seq_estm <- apply(seq_out,MARGIN=2,most_freq)
    
    write.table(paste0(as.character(con_seq_estm), collapse=","), paste0(inputPath, "/con_seq_estm.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = F)
    
  }
    
  if(dnaReference==T){
    
    ref.id<-dim(seq_out)[1]
    seq_ref<-seq_out[ref.id,]
    seq_out<-seq_out[-ref.id,]
    
    write.table(seq_out, paste0(inputPath, "/seq.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = F)
    
    
    ##con_seq_estm.txt
    write.table(paste0(as.character(seq_ref), collapse=","), paste0(inputPath, "/con_seq_estm.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = F)
    
  }

  
  write.table(keyInits, paste0(inputPath, "/parameters_key_inits.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = names(keyInits))
  
  write.table(priors, paste0(inputPath, "/parameters_priors_etc.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = names(priors))
  
  write.table(scalingFactors, paste0(inputPath, "/parameters_scaling_factors.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = names(scalingFactors))

  
  write.table(moves.inputs, paste0(inputPath, "/moves.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = names(moves.inputs))    
  
  
  #Call the main code
  res = infer.main()
  
  res$call = match.call()
  class(res) = c("infer" , "BORIS")

  # res
  return(list(epi.data = epi.data,
              moves.inputs = moves.inputs,
              parsAux = parsAux,
              keyInits = keyInits,
              priors = priors,
              scalingFactors = scalingFactors,
              seed = res$seed,
              accTable = accTable,
              t.sample = res$t_sample,
              inputPath = inputPath,
              outputPath = outputPath,
              dnaPath = dnaPath))
}