#!/bin/zsh
#SBATCH --job-name=tma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=/home/wooju.chung/TMA/output_tma/Output_tma_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=wooju.chung@ucalgary.ca 

echo "Contents of this script:"
cat "${0}"  # Prints the contents of the script
eval "$(/home/wooju.chung/software/miniconda3/bin/conda shell.bash hook)"
conda activate enel645

JOB_ID=$SLURM_JOB_ID  # job id for environment variable

python test.py "$JOB_ID"