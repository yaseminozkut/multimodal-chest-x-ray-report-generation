#!/bin/bash
#SBATCH --job-name=mllama_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "---"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mllama

# Go to project directory
cd /research/nfs_yilmaz_15/yasemin/other_projects/old_projects/mllama

# Run training
python train.py

echo "---"
echo "End time: $(date)"
