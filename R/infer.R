#' @export
#' 
infer <- function(covariates = NULL,
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
                  debug = NULL) UseMethod("infer")