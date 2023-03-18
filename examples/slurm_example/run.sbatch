#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3  # <NUMBER OF CHECKPOINTS>
#SBATCH --job-name=trak

MODEL_ID=$SLURM_ARRAY_TASK_ID

python featurize_and_score.py --model_id $MODEL_ID
