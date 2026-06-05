#!/bin/bash
#SBATCH -N 3
#SBATCH -n 24
##SBATCH -c 2
#SBATCH -C gpu
##SBATCH -G 4
#SBATCH -q debug
#SBATCH -J spmd_test
#SBATCH -t 00:05:00
#SBATCH -A m4106

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PLACES="${OMP_PLACES:-threads}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"

ERF_EXE="${ERF_EXE:-./erf_exec}"
ERF_INPUTS="${ERF_INPUTS:-inputs_BndryReg}"
ERF_RANKS="${ERF_RANKS:-$((4 * ${SLURM_NNODES:-1}))}"
ERF_LOCAL_RANKS="${ERF_LOCAL_RANKS:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NTASKS="${SLURM_NTASKS:-$((128 * ${SLURM_NNODES:-1}))}"

export ERF_EXE ERF_INPUTS ERF_RANKS ERF_LOCAL_RANKS GPUS_PER_NODE

if (( ERF_RANKS <= 0 || ERF_RANKS >= NTASKS )); then
    echo "ERF_RANKS must satisfy 0 < ERF_RANKS < NTASKS" >&2
    exit 1
fi

if (( ERF_LOCAL_RANKS < 0 )); then
    echo "ERF_LOCAL_RANKS must be non-negative" >&2
    exit 1
fi

srun -n "${NTASKS}" --cpu-bind=cores --gpu-bind=none --gpus-per-node="${GPUS_PER_NODE}" \
    bash -lc '
        set -euo pipefail
        if (( SLURM_LOCALID < ERF_LOCAL_RANKS )); then
            export MPICH_OFI_NIC_POLICY=GPU
            export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID % GPUS_PER_NODE))
            exec "${ERF_EXE}" "${ERF_INPUTS}" -- "${ERF_RANKS}"
        else
            export CUDA_VISIBLE_DEVICES=""
            export HIP_VISIBLE_DEVICES=""
            export ROCR_VISIBLE_DEVICES=""
            unset ZE_AFFINITY_MASK
            exec "${ERF_EXE}" "${ERF_INPUTS}" -- "${ERF_RANKS}"
        fi
    '
