#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=charleskimball2022@u.northwestern.edu
#SBATCH --account=b1119
#SBATCH --partition=posydon-priority
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --job-name="combine"
#SBATCH --output=output/combine_and_reset.out
#SBATCH --error=output/combine_and_reset.err

OUTFILE=reset_pop.h5
cwd=$(pwd)

srun python ../combine_and_reset_to_CC1.py --infolder $cwd --outfile cwd/$OUTFILE
