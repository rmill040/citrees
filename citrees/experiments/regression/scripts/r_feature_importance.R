#!usr/bin/env Rscript

set.seed(1718)
library(jsonlite)
library(party) # Grab the package written in C

# Constants
ALPHAS    <- c(.01, .05, .95, .99)
DATA_SETS <- c('coepra1', 'coepra2', 'coepra3', 'residential', 'comm_violence',
               'community_crime', 'facebook', 'imports-85')

# Loop over data sets
args    <- commandArgs(trailingOnly=TRUE)
results <- list()
for(name in DATA_SETS){
  
  print(paste("[info] Data set : ", name, sep=""))
  
  results[[name]] <- list()
  
  # Load data
  path <- paste(args[1], '/r_data/', name, '.csv', sep='')
  data <- read.table(path, sep=',', header=TRUE)

  # Loop over different alphas for cforest
  for(alpha in ALPHAS){
    
    print(paste("[info] Alpha : ", alpha, sep=""))
    
    # Fit model on data
    reg <- cforest(y ~ ., 
                   data=data, 
                   controls=cforest_control(teststat="max",
                                            mtry=as.integer(sqrt(ncol(data)-1)), 
                                            testtype="Univariate", 
                                            mincriterion=1-alpha, 
                                            ntree=200)
                   )
    
    # Calculate feature importances and rank from best to least
    fi                                   <- varimp(reg, conditional=FALSE)
    results[[name]][[toString(1-alpha)]] <- order(fi, decreasing=TRUE) - 1
  }
  
  # Write to .json file
  path <- paste(dirname(args[1]), '/data/r_regression.json', sep='')
  write_json(toJSON(results), path)
}
