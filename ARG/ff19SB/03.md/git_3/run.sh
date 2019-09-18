#!/bin/bash

export AMBERHOME=/mnt/raidc2/kbelfon/amber_git19/amber
export LD_LIBIRARY=${LD_LIBRARY_PATH}:$AMBERHOME/lib
export CUDA_HOME=/usr/local/cuda-8.0

$AMBERHOME/bin/pmemd -O -i md.in -p ../../01.leap/ARGdipeptide.top -c ../../02.min/min.rst -o md.out -x md.x -inf mdinfo -r md.rst
