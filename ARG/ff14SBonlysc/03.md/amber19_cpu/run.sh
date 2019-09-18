#!/bin/bash

export AMBERHOME=/mnt/raidc2/ywang/amber18
export LD_LIBIRARY=${LD_LIBRARY_PATH}:$AMBERHOME/lib

$AMBERHOME/bin/pmemd -O -i md.in -p ../../01.leap/ARGdipeptide.top -c ../../02.min/min.rst -o md.out -x md.x -inf mdinfo -r md.rst
