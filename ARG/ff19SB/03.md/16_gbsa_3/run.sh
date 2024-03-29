#!/bin/bash

export AMBERHOME=/cavern/ctian/amber16testGBSAfinal_compilcuda8.0_original_withCMAP
export LD_LIBIRARY=${LD_LIBRARY_PATH}:$AMBERHOME/lib
export CUDA_HOME=/usr/local/cuda-8.0

$AMBERHOME/bin/pmemd -O -i md.in -p ../../01.leap/ARGdipeptide.top -c ../../02.min/min.rst -o md.out -x md.x -inf mdinfo -r md.rst
