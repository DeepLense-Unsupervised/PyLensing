#!/bin/bash
echo PyAutoLens installation - Mac
cd ~
brew cask install anaconda
export PATH="/usr/local/anaconda3/bin:$PATH"
conda create -n autolens python=3.7 anaconda
conda activate autolens
conda install -c conda-forge multinest
pip install autolens
pip3 install autolens
# Change the below lines to your working directory
git clone https://github.com/Jammy2211/autolens_workspace
export WORKSPACE=/Users/pranath/autolens_workspace
export PYTHONPATH=/Users/pranath/autolens_workspace
echo Setup is Complete
