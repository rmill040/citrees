#!/usr/bin/env bash

##
# Runs feature importance experiments and stores results in mongodb
##

# Create database directory and start MongoDB
echo "Setting up mongodb"
mkdir ~/db
touch ~/db/mongodb.log
mongod --fork --dbpath ~/db --logpath ~/db/mongodb.log

# Run python script
echo "Beginning experiment"
anaconda2/bin/python citrees/citrees/experiments/regression/scripts/regression_experiment.py

# Saves results from MongoDB to .json file
echo "Creating .json file of experiment results"
mongoexport --db fi --collection regression --out regression.json

# Kill mongod application
echo "Stopping MongoDB process"
pkill mongod

# Save results to google cloud storage
echo "Saving results to GCP storage"
gsutil cp *.json gs://experiment-results

# Stop instance
echo "Script finished, stopping current instance"
gcloud compute instances stop reg-experiment -q --zone us-east4-c