#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=20
#SBATCH --time=06:00:00
#SBATCH --job-name=GAU
#SBATCH --output=jobs/outputs/output_%x_%j.out
#SBATCH --error=jobs/errors/error_%x_%j.err

module load anaconda
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate /home/FYP/muha0262/.conda/envs/gau-cuda
module load cuda/11.8
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python executable: $(which python)"
python --version

echo -e "\n=== PyTorch CUDA Verification ==="
python -c "import torch; print('torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo -e "\n=== GPU Verification ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi || echo "nvidia-smi command failed or no GPU detected!"
else
    echo "nvidia-smi not found. Cannot verify GPU presence."
fi

echo -e "\n=== CUDA Version Check ==="
nvidia-smi

# Verify torch and CUDA
python -c "import torch; print('torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo -e "\n=== Running GAU ==="
echo "Started at: $(date)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32

cmd="python -m main"

echo "Running: $cmd"
eval "$cmd"
if [ $? -ne 0 ]; then
    echo "Error: main.py failed"
    exit 1
fi

echo "Finished at: $(date)"
