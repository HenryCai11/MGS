#!/bin/bash
#SBATCH --job-name=mgs-qwen14b-math-chat
#SBATCH --account=def-xiye17
#SBATCH --partition=gpubase_bynode_b4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/henrycai/MGS/scripts/slurm/logs/train_mgs_14b_%j.out
#SBATCH --error=/scratch/henrycai/MGS/scripts/slurm/logs/train_mgs_14b_%j.err

# Modular Gradient Surgery (MGS) GRPO training run on Qwen2.5-14B-Prompted
# with math (simplerl_level3to5/train_clean.parquet) + chat (wildchat-if)
# mixed 1:1 via MultiSourceRLHFDataset + WeightedRandomSampler.
#
# Multi-node Ray bootstrap: head node runs `ray start --head`, worker nodes
# join via `ray start --address=<head>:6379`, then we kick off the training
# entry point with `python -m verl.trainer.main_ppo_mixed`.

set -euo pipefail
mkdir -p /scratch/henrycai/MGS/scripts/slurm/logs

module load cuda/13.2

# -------- Paths --------
REPO_DIR="/scratch/henrycai/MGS"
PY_ENV="/scratch/henrycai/qwen-serve-env"
PYTHON="$PY_ENV/bin/python"
CUDA13_LIB="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/13.2.0/lib64"
VERL_DIR="/scratch/henrycai/verl"

# Models / data
MODEL_PATH="${MODEL_PATH:-/scratch/henrycai/models/Qwen2.5-14B-Prompted}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-/scratch/henrycai/models/Skywork-Reward-V2-Llama-3.1-8B}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: prompted model missing at $MODEL_PATH. Run 01_make_prompted.slurm first." >&2
    exit 1
fi
if [ ! -d "$REWARD_MODEL_PATH" ]; then
    echo "ERROR: reward model missing at $REWARD_MODEL_PATH. Run 00_download_models.slurm first." >&2
    exit 1
fi
if [ ! -f "$REPO_DIR/mgs/training/grpo/data/simplerl_level3to5/train_clean.parquet" ]; then
    echo "ERROR: math train_clean.parquet missing." >&2
    exit 1
fi
if [ ! -f "$REPO_DIR/mgs/training/grpo/data/wildchat-if_train.parquet" ]; then
    echo "ERROR: chat wildchat-if_train.parquet missing." >&2
    exit 1
fi

# -------- Runtime env --------
export PATH="$PY_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA13_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$VERL_DIR:$REPO_DIR:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_HCA=mlx5
export NCCL_NVLS_ENABLE=1

# WANDB — set WANDB_API_KEY in your shell before sbatch if you want logging,
# otherwise wandb falls back to offline mode under WANDB_MODE=offline.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -z "$WANDB_API_KEY" ]; then
    export WANDB_MODE=offline
fi
export WANDB_DIR="$REPO_DIR/checkpoints/wandb"
mkdir -p "$WANDB_DIR"

# -------- Cluster topology --------
NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=4
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))

nodes_array=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
HEAD_NODE=${nodes_array[0]}
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
RAY_PORT=6379
DASHBOARD_PORT=8265

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes ($NNODES): ${nodes_array[*]}"
echo "Head node:     $HEAD_NODE ($HEAD_IP)"
echo "Total GPUs:    $TOTAL_GPUS"
echo "Model:         $MODEL_PATH"
echo "Reward model:  $REWARD_MODEL_PATH"
echo "Repo:          $REPO_DIR"
echo "Python:        $($PYTHON --version)"
echo "========================================"

cd "$REPO_DIR"

# -------- Start Ray head --------
echo "Starting Ray head on $HEAD_NODE ..."
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --gres=gpu:$GPUS_PER_NODE \
    bash -c "$PY_ENV/bin/ray start --head \
        --node-ip-address=$HEAD_IP \
        --port=$RAY_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$DASHBOARD_PORT \
        --num-gpus=$GPUS_PER_NODE \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --block" &
HEAD_PID=$!
sleep 30

# -------- Start Ray workers --------
for ((i=1; i<NNODES; i++)); do
    node=${nodes_array[$i]}
    echo "Starting Ray worker on $node ..."
    srun --nodes=1 --ntasks=1 -w "$node" --gres=gpu:$GPUS_PER_NODE \
        bash -c "$PY_ENV/bin/ray start \
            --address=$HEAD_IP:$RAY_PORT \
            --num-gpus=$GPUS_PER_NODE \
            --num-cpus=$SLURM_CPUS_PER_TASK \
            --block" &
    sleep 5
done

echo "Waiting for Ray cluster to come up ..."
sleep 30
$PY_ENV/bin/ray status --address="$HEAD_IP:$RAY_PORT" || true

# -------- Launch trainer --------
RUN_NAME="mgs_qwen14b_math_chat_1to1_think_${SLURM_JOB_ID}"

# Address Ray at the head; main_ppo_mixed uses ray.init() which will pick up
# RAY_ADDRESS so the trainer task is scheduled on the cluster we just brought up.
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"

echo "Launching MGS GRPO training ..."
$PYTHON -m verl.trainer.main_ppo_mixed \
    --config-path mgs/training/grpo/configs \
    --config-name mgs_mixed_grpo_qwen14b_math_chat_think \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    critic.model.path="$MODEL_PATH" \
    reward_model.model.path="$REWARD_MODEL_PATH" \
    reward_model.longcot_config="$MODEL_PATH/longcot_config.json" \
    data.train_files=mgs/training/grpo/data/wildchat-if_train.parquet \
    data.val_files=mgs/training/grpo/data/wildchat-if_val.parquet \
    'data.source_proportions={math:0.5,chat:0.5}' \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.project_name=rlvr_rlmt_thinking \
    trainer.experiment_name="$RUN_NAME" \
    trainer.default_local_dir="/scratch/henrycai/MGS/checkpoints/rlvr_rlmt_thinking/$RUN_NAME" \
    trainer.resume_mode=auto \
    "$@"

echo "Training complete. Stopping Ray ..."
$PY_ENV/bin/ray stop || true
wait
