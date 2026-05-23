# MGS — Qwen2.5-14B math + chat training on FIR (Alliance Canada)

End-to-end SLURM pipeline to reproduce MGS GRPO training on
`Qwen/Qwen2.5-14B`, mixing math (`simplerl_level3to5/train_clean.parquet`)
with chat (`wildchat-if_train.parquet`) at a 1:1 ratio.

## Cluster assumptions

- Account: `def-xiye17` (def-xiye17_gpu)
- Partition: `gpubase_bynode_b4` (3-day GPU partition, 4×H100 80GB per node)
- Python env: `/scratch/henrycai/qwen-serve-env` (torch 2.11, vLLM 0.20.2, verl, datasets)
- VERL source: `/scratch/henrycai/verl`

If any of these change, edit the three scripts in this folder.

## Pipeline

```
00_download_models.slurm   # CPU, ~30 min: hf-download Qwen2.5-14B + Skywork RM
01_make_prompted.slurm     # 1×H100, ~10 min: apply RLMT think-template to base
02_train_mgs_14b.slurm     # 4×4 H100s, up to 3 days: GRPO MGS training
```

### Step 0 — download base model + reward model

```
sbatch /scratch/henrycai/MGS/scripts/slurm/00_download_models.slurm
```

Writes to:
- `/scratch/henrycai/models/Qwen2.5-14B`
- `/scratch/henrycai/models/Skywork-Reward-V2-Llama-3.1-8B`

### Step 1 — convert base into a "prompted" think-template variant

Runs the RLMT `convert_base_to_prompted_model.py` script (mirrored at
`mgs/training/grpo/scripts/data/convert_base_to_prompted_model.py`). The
output gets a chat template instructing the model to reason inside
`<think>...</think>` and answer inside `<response>...</response>`, plus the
`longcot_config.json` that the reward model worker expects.

```
sbatch /scratch/henrycai/MGS/scripts/slurm/01_make_prompted.slurm
```

Writes `/scratch/henrycai/models/Qwen2.5-14B-Prompted`.

### Step 2 — MGS GRPO training (math + chat, 1:1)

```
# optional: enable wandb online logging
export WANDB_API_KEY=...

sbatch /scratch/henrycai/MGS/scripts/slurm/02_train_mgs_14b.slurm
```

What this does:

- 4 nodes × 4 H100s = 16 GPUs, time limit 3 days
- Brings up a Ray cluster across all 4 nodes (head + 3 workers)
- Runs `python -m verl.trainer.main_ppo_mixed` with config
  `mgs/training/grpo/configs/mgs_mixed_grpo_qwen14b_math_chat_think.yaml`
- `MultiSourceRLHFDataset` loads both parquets, `WeightedRandomSampler`
  enforces `{math:0.5, chat:0.5}` per batch
- Reward routing: math goes through the rule-based math verifier; chat
  goes through the Skywork-Reward-V2-Llama-3.1-8B reward model worker
- Checkpoints land under
  `/scratch/henrycai/MGS/checkpoints/rlvr_rlmt_thinking/<run>`
- `trainer.resume_mode=auto` means re-submitting the same job picks up
  the latest checkpoint

## Tuning knobs (override via extra sbatch args)

`02_train_mgs_14b.slurm` forwards any extra args after `--` to the python
entry point, so you can override config from the command line:

```
sbatch 02_train_mgs_14b.slurm -- \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    'data.source_proportions={math:0.7,chat:0.3}'
```

Important defaults already set for 14B:

- `actor.fsdp_config.param_offload=True`, `optimizer_offload=True`
- `ref.fsdp_config.param_offload=True`
- `rollout.tensor_model_parallel_size=2` (8 vLLM ranks across the cluster)
- `rollout.n=8` rollouts per prompt
- `data.train_batch_size=64`, `ppo_mini_batch_size=64`
- `data.max_prompt_length=896`, `data.max_response_length=4096`

If you OOM, try:
- bump `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` down (already 1)
- increase `tensor_model_parallel_size` to 4
- reduce `rollout.gpu_memory_utilization` to 0.3

## Validation

Validation is currently math-only (`simplerl_level3to5/test_clean.parquet`)
— the chat val parquet is wired in but commented out in `main_ppo_mixed.py`
(no reward signal is computed for the chat val side without a judge).

## Where things live in the repo

- Conversion script: `mgs/training/grpo/scripts/data/convert_base_to_prompted_model.py`
- Mixed dataset:     `mgs/training/grpo/verl/utils/dataset/mixed_dataset.py`
- 14B config:        `mgs/training/grpo/configs/mgs_mixed_grpo_qwen14b_math_chat_think.yaml`
- Trainer entry:     `mgs/training/grpo/verl/trainer/main_ppo_mixed.py`
- MGS actor patch:   `mgs/training/grpo/verl/workers/dp_actor.py`
