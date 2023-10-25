#!/bin/bash
#SBATCH --job-name=jl_unit_test
#SBATCH --partition=high-priority
#SBATCH hetjob
#SBATCH --output=/mnt/xfs/home/alaakh/installs/temp/trak_fixes/tests/test_jl_gpu_compatibility/%u-%x-%j.log

CODE_PATH="/mnt/xfs/home/alaakh/installs/temp/trak_fixes/tests/test_jl_gpu_compatibility"
cd $CODE_PATH

export PYTHONPATH="${PYTHONPATH}:${CODE_PATH}"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CREATE_PROJ_TEST="test_jl_gpu_compatibility.py::test_create_proj"
VERIFY_PROJ_TEST="test_jl_gpu_compatibility.py::test_same_proj"

# Component for a100 GPU
#SBATCH --nodes=1
#SBATCH --nodelist="deep-chungus-[1-5,7-11]"
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=high-priority

env GPU_NAME="A100" \
srun --ntasks=1 \
     python -m pytest $CREATE_PROJ_TEST

# Component for h100 GPU
#SBATCH hetjob
#SBATCH --nodes=1
#SBATCH --nodelist="deep-h-[1-3]"
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=high-priority

env GPU_NAME="H100" \
srun --ntasks=1 \
     python -m pytest $CREATE_PROJ_TEST

srun --ntasks=1 \
    python -m pytest $VERIFY_PROJ_TEST

rm "${CODE_PATH}/A100.pt"
rm "${CODE_PATH}/H100.pt"

echo "done"