#!/usr/bin/env bash

##
# Runs feature importance experiments and stores results in mongodb
##

# Start mongodb on default port 27017
pkill mongod
echo "Starting mongodb on localhost port 27017"
mongod --fork --dbpath ~/db --logpath ~/db/mongodb.log

# Run python script to generate results
echo "Running regression experiment"
python regression_experiment.py

# Kill mongodb process
echo "Script finished, killing monogodb process"
pkill mongod
