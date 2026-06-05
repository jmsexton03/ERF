#!/bin/bash
# Modulo 4 ensures it cycles 0, 1, 2, 3 based on local rank
export MPICH_OFI_NIC_POLICY=GPU
export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID % 4))
echo "CUDA is $CUDA_VISIBLE_DEVICES"
exec "$@"
