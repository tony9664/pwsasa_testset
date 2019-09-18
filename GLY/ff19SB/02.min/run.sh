#!/bin/bash

export AMBERHOME=/opt/amber
export LD_LIBIRARY=${LD_LIBRARY_PATH}:$AMBERHOME/lib

$AMBERHOME/bin/pmemd -O -i min.in -p ../01.leap/GLYdipeptide.top -c ../01.leap/GLYdipeptide.crd -o min.out -x min.x -inf mininfo -r min.rst
