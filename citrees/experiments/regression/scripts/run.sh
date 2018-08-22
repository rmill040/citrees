#!/usr/bin/env bash

##
# Runs feature importance experiments and stores results in mongodb
##

# Start mongodb on default port 27017
mongod --fork \
       --dbpath /Users/R2/Documents/Research/citrees/citrees/experiments/regression/data/db \
       --logpath /Users/R2/Documents/Research/citrees/citrees/experiments/regression/data/db/mongodb.log

# Run python script to generate results
echo "ELLO"

# Run python script to analyze results
echo "ELLO"

# Kill mongodb process
#pkill mongod