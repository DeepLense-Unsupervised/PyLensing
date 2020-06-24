#!/bin/bash
echo PyAutoLens installation - Mac
cd ~
brew cask install anaconda
export PATH="/usr/local/anaconda3/bin:$PATH"
conda create -n autolens python=3.7 anaconda
conda activate autolens
conda install -c conda-forge multinest
pip install autolens==0.46.2
pip3 install autolens==0.46.2
git clone https://github.com/Jammy2211/autolens_workspace
cd autolens_workspace/
git checkout 5acb22d22539ccac3328d65d64e29c66aaa24fc7
cd ..
# Change the below lines to your working directory
export WORKSPACE=/Users/pranath/autolens_workspace
export PYTHONPATH=/Users/pranath/autolens_workspace
echo Setup is Complete
