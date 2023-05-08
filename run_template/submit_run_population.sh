#!/bin/bash
#SBATCH --account=b1094 ## <-- EDIT THIS TO BE YOUR ALLOCATION 
#SBATCH --partition=ciera-std ## <-- EDIT THIS TO BE YOUR QUEUE NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-100%100
#SBATCH --job-name=pop_default
#SBATCH --output=output/run_population_%A_%a.out
#SBATCH --error=output/run_population_%A_%a.err


cwd=$(pwd)

export PATH_TO_POSYDON=/projects/b1119/POSYDON/
mkdir batches/batch_${SLURM_ARRAY_TASK_ID}
cd batches/batch_${SLURM_ARRAY_TASK_ID}

mpiexec -n ${SLURM_NTASKS} python ../../../run_population.py --infolder $cwd --outfile population_${SLURM_ARRAY_TASK_ID}.h5

