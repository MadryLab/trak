#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=trak

MODEL_ID=$SLURM_ARRAY_TASK_ID
CHECKPOINT="/path/to/checkpoints/ckpt_${MODEL_ID}.pt"

python get_scores.py
