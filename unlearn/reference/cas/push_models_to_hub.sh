#!/bin/bash

#SBATCH --job-name=push_to_hub                     				                    	# Job name
#SBATCH --output=/home/a5a/scasper.a5a/owsg/outfiles/push_to_hub%A_%a.out            	 # Standard output and error log
#SBATCH --array=0                  					                       # Array range
#SBATCH --time=24:00:00                  					                     # Time limit hrs:min:sec
#SBATCH --reservation=aisi               				 	                        # Reservation
#SBATCH --gpus=4                         					                    # Number of GPUs

echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"

cd /lus/lfs1aip1/home/a5k/scasper.a5a/
source miniforge3/bin/activate
conda activate owsg
cd /lus/lfs1aip1/home/a5k/scasper.a5a/owsg

python push_models_to_hub.py
