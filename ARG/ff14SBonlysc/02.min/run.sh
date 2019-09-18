#!/bin/bash

$AMBERHOME/bin/pmemd -O -i min.in -p ../01.leap/ARGdipeptide.top -c ../01.leap/ARGdipeptide.crd -o min.out -x min.x -inf mininfo -r min.rst
