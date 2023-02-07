#!/usr/bin/env bash

# Install dependencies
here=$(pwd)
sudo yum update -y
sudo yum install gcc openssl-devel bzip2-devel libffi-devel -y 

# Build Python 3.9 from source
cd /opt
sudo wget https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz
sudo tar xzf Python-3.9.16.tgz
cd Python-3.9.16
sudo ./configure --enable-optimizations --enable-loadable-sqlite-extensions
sudo make altinstall -j $(nproc)
sudo rm -f /opt/Python-3.9.16.tgz 

# Install poetry and additional dependencies
cd $here
curl -sSL https://install.python-poetry.org | python3.9 -
poetry install -E experiments