#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name test
#SBATCH --partition=780,980,1080
hostname
echo $CUDA_VISIBLE_DEVICES
pwd

export AMBERHOME=/mnt/raidc2/ywang/amber18
export LD_LIBIRARY=${LD_LIBRARY_PATH}:$AMBERHOME/lib
export CUDA_HOME=/usr/local/cuda-8.0

$AMBERHOME/bin/pmemd.cuda -O -i md.in -p ../../01.leap/ARGdipeptide.top -c ../../02.min/min.rst -o md.out -x md.x -inf mdinfo -r md.rst
