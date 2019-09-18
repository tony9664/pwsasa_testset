#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name test
#SBATCH --partition=780,980,2080,mix,prometheus,minotaur

hostname
echo $CUDA_VISIBLE_DEVICES
pwd

export AMBERHOME=/mnt/raidc2/kbelfon/amber_new/amber
source $AMBERHOME/amber.sh
export CUDA_HOME=/usr/local/cuda-9.2

$AMBERHOME/bin/pmemd.cuda_SPFP -O -i md.in -p ../../01.leap/GLYdipeptide.top -c ../../02.min/min.rst -o md.out -x md.x -inf mdinfo -r md.rst
