#!/bin/bash

export AMBERHOME=/mnt/raidc2/kbelfon/amber_git19/amber
export LD_LIBIRARY=${LD_LIBRARY_PATH}:$AMBERHOME/lib

$AMBERHOME/bin/pmemd -O -i md.in -p ../../01.leap/GLYdipeptide.top -c ../../02.min/min.rst -o md.out -x md.x -inf mdinfo -r md.rst
