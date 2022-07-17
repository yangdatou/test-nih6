#!/bin/bash
#SBATCH --time=60
#SBATCH --job-name=test
#SBATCH --partition=debug
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=28       # cpu-cores per task
#SBATCH --mem=0

module purge
module load gcc/9.2.0
module load binutils/2.26
module load cmake-3.6.2 

export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28

source /home/yangjunjie/intel/oneapi/setvars.sh --force;
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

export PYTHONPATH=/home/yangjunjie/packages/pyscf/pyscf-main/
export PYTHONPATH=/home/yangjunjie/packages/libdmet/libdmet_solid-model/:$PYTHONPATH

export TMPDIR=/scratch/global/yangjunjie/${SLURM_JOB_NAME}/$SLURM_JOBID/
export PYSCF_TMPDIR=/scratch/global/yangjunjie/${SLURM_JOB_NAME}/$SLURM_JOBID/

mkdir -p $PYSCF_TMPDIR

rm ./data/*
rm ./xyz/*
rm ./log/*
python main.py
