#' @export
#' 
sim <- function(epi.inputs = NULL, 
                moves.inputs = NULL, 
                parsKey = NULL,
                parsAux = NULL, 
                inputPath = NULL, 
                outputPath = NULL,
                debug = NULL) UseMethod("sim")
