#!/usr/bin/env bash

##
# Runs feature importance experiments and stores results in mongodb
##

# Run python script
echo "Beginning experiment"
anaconda2/bin/python citrees/citrees/experiments/classifier/scripts/classifier_experiment.py

# Saves results from MongoDB to .json file
echo "Creating .json file of experiment results"
mongoexport --db fi --collection classifier --out classifier.json

# Kill mongod application
echo "Stopping MongoDB process"
pkill mongod

# Save results to google cloud storage
echo "Saving results to GCP storage"
gsutil cp *.json gs://experiment-results

# Detach from tmux session and stop instance
tmux detach
echo "Script finished, stopping current instance"
gcloud compute instances stop instance-1 -q --zone us-east1-b
