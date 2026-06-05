#!/bin/bash
#SBATCH -N 1
#SBATCH -n 128 # 12 ERF tasks + 120 NoahMP tasks
##SBATCH -c 2 # 2 hyperthreads per MPI task (leaves some room for OS)
#SBATCH -C gpu
##SBATCH -G 12 # Ask for all 12 GPUs on the 3 nodes
#SBATCH -q debug
#SBATCH -J mpmd_test
#SBATCH -t 00:05:00
#SBATCH -A m4106

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
#export GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

# The wrappers handle the GPU bindings, so srun can be simple
srun -n 128 --multi-prog --cpu-bind=cores --gpu-bind=none --gpus-per-node=4 ./mpmd_gpu_cpu.conf
