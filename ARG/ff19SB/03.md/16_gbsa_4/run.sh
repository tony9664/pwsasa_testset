#!/bin/bash

export AMBERHOME=/mnt/raidc2/ctian/amber16testGBSAfinal_compilcuda8.0
export LD_LIBIRARY=${LD_LIBRARY_PATH}:$AMBERHOME/lib

$AMBERHOME/bin/pmemd -O -i md.in -p ../../01.leap/ARGdipeptide.top -c ../../02.min/min.rst -o md.out -x md.x -inf mdinfo -r md.rst
