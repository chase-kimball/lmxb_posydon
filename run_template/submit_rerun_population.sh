#!/bin/bash
#SBATCH --account=b1119 ## <-- EDIT THIS TO BE YOUR ALLOCATION 
#SBATCH --partition=posydon-priority## <-- EDIT THIS TO BE YOUR QUEUE NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --array=1-500%500
#SBATCH --job-name=SN_15_50
#SBATCH --output=output/rerun_population_%A_%a.out
#SBATCH --error=output/rerun_population_%A_%a.err

infile=reset_pop.h5

export PATH_TO_POSYDON=/projects/b1119/POSYDON/
mkdir kickbatches/batch_${SLURM_ARRAY_TASK_ID}
cp $infile kickbatches/batch_${SLURM_ARRAY_TASK_ID}
cd kickbatches/batch_${SLURM_ARRAY_TASK_ID}
python ../../../rerun_population.py --infile $infile --outfile rerun_${SLURM_ARRAY_TASK_ID}.h5
rm $infile

