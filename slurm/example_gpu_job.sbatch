#!/bin/bash
# Example Slurm job. Usage:
#   sbatch --export=ALL,CMD="python scripts/eval_eventhdr.py --generator ... --shader ... --data_root ... --output_dir outputs/eval" slurm/example_gpu_job.sbatch
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=hive
#SBATCH --output=logs/%x_%j.out
export HF_HOME=/path/to/hf_cache        # where the SVD weights were downloaded
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
source /path/to/venv/bin/activate
mkdir -p logs
echo "CMD: $CMD"; eval "$CMD"
