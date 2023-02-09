#!/usr/bin/env bash

# Install dependencies
here=$(pwd)
yum update -y
yum install gcc openssl-devel bzip2-devel libffi-devel tmux -y 

# Build Python 3.9 from source
cd /opt
wget https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz
tar xzf Python-3.9.16.tgz
cd Python-3.9.16
./configure --enable-optimizations --enable-loadable-sqlite-extensions
make altinstall -j $(nproc)
rm -f /opt/Python-3.9.16.tgz 

# Install poetry and additional Python dependencies
cd $here
curl -sSL https://install.python-poetry.org | python3.9 -
cd ../..
/root/.local/bin/poetry install --with dev,experiments