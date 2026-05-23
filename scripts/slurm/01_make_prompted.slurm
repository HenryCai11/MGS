#!/bin/bash
#SBATCH --job-name=mgs-make-prompted
#SBATCH --account=def-xiye17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/henrycai/MGS/scripts/slurm/logs/make_prompted_%j.out
#SBATCH --error=/scratch/henrycai/MGS/scripts/slurm/logs/make_prompted_%j.err

# Apply the RLMT "prompted think-template" wrapping
# (https://github.com/princeton-pli/RLMT/blob/main/training/sft_dpo/utils/src/convert_base_to_prompted_model.py)
# to Qwen/Qwen2.5-14B so it can be used as the zero-style starting point for
# MGS. Output goes to /scratch/henrycai/models/Qwen2.5-14B-Prompted with
# stop.txt and longcot_config.json next to the weights.

set -euo pipefail
mkdir -p /scratch/henrycai/MGS/scripts/slurm/logs

module load cuda/13.2

REPO_DIR="/scratch/henrycai/MGS"
PYTHON="/scratch/henrycai/qwen-serve-env/bin/python"

MODEL_SRC="/scratch/henrycai/models/Qwen2.5-14B"
MODEL_DST="/scratch/henrycai/models/Qwen2.5-14B-Prompted"

if [ ! -d "$MODEL_SRC" ]; then
    echo "ERROR: $MODEL_SRC missing. Run 00_download_models.slurm first." >&2
    exit 1
fi

echo "========================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python:    $($PYTHON --version)"
echo "Source:    $MODEL_SRC"
echo "Dest:      $MODEL_DST"
echo "========================================"

cd "$REPO_DIR"

"$PYTHON" mgs/training/grpo/scripts/data/convert_base_to_prompted_model.py \
    --model_path "$MODEL_SRC" \
    --out_path   "$MODEL_DST" \
    --test_generation

echo "Done. Listing $MODEL_DST:"
ls -lh "$MODEL_DST" | head
cat "$MODEL_DST/longcot_config.json"
