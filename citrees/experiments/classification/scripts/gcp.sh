#!/usr/bin/env bash

##############################################
# Script to run classifier experiment on GCP #
##############################################

# gcloud compute scp --zone us-east1-b ~/Documents/Research/citrees/citrees/experiments/classifier/scripts/gcp.sh instance-1:~/
# gcloud compute ssh --zone us-east1-b instance-1

# Update settings
echo "Updating settings"
sudo apt-get update

# Install Mongodb
echo "Installing MongoDB"
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 9DA31620334BD75D9DCB49F368818C72E52529D4
echo "deb http://repo.mongodb.org/apt/debian stretch/mongodb-org/4.0 main" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.0.list
sudo apt-get update
sudo apt-get install -y --allow-unauthenticated mongodb-org

# Install tmux
echo "Installing tmux"
sudo apt-get -y install tmux

# Install Git
echo "Installing Git"
sudo apt-get -y install git

# Install GCC compiler
echo "Installing gcc compiler"
sudo apt-get -y install gcc

# Install Anaconda 2
echo "Installing Anaconda2"
sudo apt-get -y install bzip2
wget https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh
bash Anaconda2-5.2.0-Linux-x86_64.sh -b

# Update pip and install extra packages
echo "Upgrading pip and install extra required packages"
anaconda2/bin/pip install --upgrade pip
anaconda2/bin/pip install pymongo joblib

# Clone citrees
echo "Cloning citrees repo"
git clone https://github.com/rmill040/citrees.git

# Clean and recompile C code
echo "Compiling code"
sudo apt-get -y install make
make clean -C citrees/citrees
make -C citrees/citrees

# Create database directory and start MongoDB
echo "Beginning experiment"
mkdir ~/db
touch ~/db/mongodb.log
mongod --fork --dbpath ~/db --logpath ~/db/mongodb.log

# Start new tmux session and run main shell script to generate results
chmod 755 run.sh
tmux new-session -d -s clf-exp "./run.sh"
