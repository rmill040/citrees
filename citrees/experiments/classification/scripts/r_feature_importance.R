#!usr/bin/env Rscript

set.seed(1718)
library(jsonlite)
library(party) # Grab the package written in C

# Constants
ALPHAS    <- c(.01, .05, .95, .99)
DATA_SETS <- c('musk', 'wine', 'orlraws10P', 'glass', 'warpPIE10P', 'warpAR10P', 'pixraw10P',
               'ALLAML', 'CLL_SUB_111', 'ORL', 'TOX_171', 'Yale',
               'vowel-context', 'gamma', 'isolet', 'letter', 'madelon',
               'page-blocks', 'pendigits', 'spam')

# Loop over data sets
args    <- commandArgs(trailingOnly=TRUE)
results <- list()
for(name in DATA_SETS){
  
  print(paste("[info] Data set : ", name, sep=""))
  
  # Load data
  path <- paste(args[1], '/r_data/', name, '.csv', sep='')

  # Skip over files not saved as .csv from python side
  if(!file.exists(path)){
    print("[info] Skipping data set")
    next
  } 
  
  data <- read.table(path, sep=',', header=TRUE)

  # Loop over different alphas for cforest
  results[[name]] <- list()
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
  path <- paste(dirname(args[1]), '/data/r_classifier.json', sep='')
  write_json(toJSON(results), path)
}
