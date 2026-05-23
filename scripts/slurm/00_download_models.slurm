#!/bin/bash
#SBATCH --job-name=mgs-dl-models
#SBATCH --account=def-xiye17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/henrycai/MGS/scripts/slurm/logs/download_%j.out
#SBATCH --error=/scratch/henrycai/MGS/scripts/slurm/logs/download_%j.err

set -euo pipefail
mkdir -p /scratch/henrycai/MGS/scripts/slurm/logs

HF_CLI="/scratch/henrycai/qwen-serve-env/bin/hf"

export HF_HOME="/scratch/henrycai/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it in your shell before sbatch." >&2
    exit 1
fi
export HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1

MODELS_DIR="/scratch/henrycai/models"
mkdir -p "$MODELS_DIR"

QWEN_DEST="$MODELS_DIR/Qwen2.5-14B"
SKYWORK_DEST="$MODELS_DIR/Skywork-Reward-V2-Llama-3.1-8B"

echo "========================================"
echo "Downloading Qwen/Qwen2.5-14B -> $QWEN_DEST"
echo "========================================"
mkdir -p "$QWEN_DEST"
"$HF_CLI" download Qwen/Qwen2.5-14B --local-dir "$QWEN_DEST"

echo "========================================"
echo "Downloading Skywork/Skywork-Reward-V2-Llama-3.1-8B -> $SKYWORK_DEST"
echo "========================================"
mkdir -p "$SKYWORK_DEST"
"$HF_CLI" download Skywork/Skywork-Reward-V2-Llama-3.1-8B --local-dir "$SKYWORK_DEST"

echo "Done."
ls -lh "$QWEN_DEST" | head
ls -lh "$SKYWORK_DEST" | head
