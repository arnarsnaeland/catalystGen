#!/bin/bash -ex
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=sm3090_devel
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --time=0-03:00:00
#SBATCH --gres=gpu:RTX3090:1 # Request 1 GPU (can increase for more)
module load Python/3.11.3-GCCcore-12.3.0
source ~/crystal-llm/bin/activate
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd ~/master-thesis
python src/pkg/main.py --model_name=7b --model_path=checkpoint-80000 --instruction_prompt="The material is Fe3O4." --num_samples=20 --ml_model_checkpoint="eq2_153M_ec4_allmd.pt" --batch_size=3