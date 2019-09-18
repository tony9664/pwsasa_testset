#!/bin/bash

export AMBERHOME=/cavern/ctian/amber16testGBSAfinal_compilcuda8.0_original_withCMAP
export LD_LIBIRARY=${LD_LIBRARY_PATH}:$AMBERHOME/lib
export CUDA_HOME=/usr/local/cuda-8.0

$AMBERHOME/bin/pmemd -O -i min.in -p ../01.leap/ARGdipeptide.top -c ../01.leap/ARGdipeptide.crd -o min.out -x min.x -inf mininfo -r min.rst
