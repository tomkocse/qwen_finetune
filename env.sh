#!/bin/bash

set -e

#load modules with
module purge
module load hosts/hopper
module load cuda/12.6.3  # this will set $CUDA_HOME
module load gnu10/10.3.0-ya
module load git/2.27.1
module load miniconda3/22.11.1-gy

source ~/.bashrc
conda activate qwen_ft
export HF_HOME=/data/share/
export HF_ENDPOINT=https://hf-mirror.com
