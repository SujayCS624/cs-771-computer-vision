#!/bin/bash
# script for setting up the environment

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge opencv
conda install numpy
conda install -c conda-forge gdown
conda install tensorboard