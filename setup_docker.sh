#!/bin/bash
echo PyAutoLens installation - Docker
git clone https://github.com/Jammy2211/autolens_workspace
cd autolens_workspace/
git checkout 5acb22d22539ccac3328d65d64e29c66aaa24fc7
cd ..
export WORKSPACE=./autolens_workspace/
export PYTHONPATH=./autolens_workspace

