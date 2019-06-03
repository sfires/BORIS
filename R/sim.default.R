#' @export
#' @useDynLib BORIS, .registration = TRUE
#' @importFrom ape read.dna
#' @importFrom utils write.table
sim.default <- function(epi.inputs = NULL, 
                        moves.inputs = NULL, 
                        parsKey = NULL,
                        parsAux = NULL, 
                        inputPath = NULL, 
                        outputPath = NULL,
                        debug = NULL){
  
  options(warn=-1)
  if ( .Platform$OS.type != "windows" ) {
    slash <- "/"
  } else {
    slash <- "\\"
  }
  
  if (is.null(epi.inputs)){
    stop("Please supply the epidemiological input data by the argument 'epi.inputs'")
  }
  
  if (is.null(moves.inputs)){
    stop("Please supply the movement (contract-tracing) input data by the argument 'moves.inputs'. If 'opt_mov' is set to 0 these data will be ignored.")
  }
  
  if (is.null(parsKey)){
    stop("Please supply the key initial values by the argument 'parsKey'")
  }
  
  if (is.null(parsAux)){
    stop("Please supply the auxiliary parameters by the argument 'parsAux'")
  }
  parsAux[1, "n"] = round(parsAux[1, "n"])
  parsAux[1, "seed"] = round(parsAux[1, "seed"])
  parsAux[1, "n_base"] = round(parsAux[1, "n_base"])
  parsAux[1, "n_seq"] = round(parsAux[1, "n_seq"])
  parsAux[1, "n_base_part"] = round(parsAux[1, "n_base_part"])
  parsAux[1, "n_index"] = round(parsAux[1, "n_index"])
  parsAux[1, "n_mov"] = round(parsAux[1, "n_mov"])

  sim.checkInputs(parsKey = parsKey, 
                  parsAux = parsAux)
  
  
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
  write.table(epi.inputs, paste0(inputPath, "/epi_in.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F,col.names = TRUE)
  
  ##moves.data
  write.table(moves.inputs, paste0(inputPath, "/mov_in.csv"), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F,col.names = TRUE)
  
  ##parameters_key.csv
  parsKey <- data.frame(parsKey)
  write.table(parsKey, file = paste0(inputPath, "/parameters_key.csv", sep=''), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = names(parsAux) )  

  ##parameters_other.csv
  parsAux <- data.frame(parsAux)
  
  if(is.null(debug)){parsAux$debug = 0}
  if(!is.null(debug)){
    if(debug == 1){parsAux$debug = 1}else{parsAux$debug = 0}
  }
  
  write.table(parsAux, file = paste0(inputPath, "/parameters_other.csv", sep=''), quote = F, sep = ",",eol = "\n", na = "NA", dec = ".", row.names = F, col.names = names(parsAux) )  
    

  #Call the main code
  res = sim.main()
  
  res$call = match.call()
  class(res) = c("sim" , "BORIS")

  # res
  return(list(epi.sim = res$epi.sim,
              moves.inputs = moves.inputs,
              parsKey = parsKey,
              parsAux = parsAux,
              infected_source = res$infected_source,
              t_sample = res$t_sample,
              sampled_perct = res$sampled_perct,
              cons_seq = res$cons_seq,
              inputPath = inputPath,
              outputPath = outputPath))
}
