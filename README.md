# Advancing General-Purpose Reasoning Models with Modular Gradient Surgery

> Anonymized code release for double-blind review.

This repository contains the implementation of **Modular Gradient Surgery (MGS)**, a framework designed to optimize Large Language Models (LLMs) across conflicting objectives—specifically balancing mathematical reasoning, general chat capabilities, and instruction following (IFEval).

## 💡 Overview

Effectiveness of different ways for training reasoning models on multiple domains. Naive strategies, such as sequential RL training (Sequential RL), or mixing different domains in the same batch (Mixed RL), often result in limited performance across domains. We propose Modular Gradient Surgery (MGS), which resolves conflicting gradients at the module level and achieves the best multi-domain performance.

## ⚙️ Setup

You may use your preferred package manager (`uv`, `conda`, or `venv`). Note: Install PyTorch first to ensure all subsequent dependencies are resolved correctly.

> Note: This repository is fully compatible with existing RLMT environments. If you already have one configured, you may use it directly without further setup.

### 1. Environment Creation

#### Option A: Using uv
```bash
# Create and activate environment
uv venv mgs --python 3.10
source mgs/bin/activate

# Install PyTorch and dependencies
uv pip install torch torchvision torchaudio
uv pip install -r requirements.txt
```
#### Option B: Using conda
```bash
# Create and activate environment
conda create -n mgs python=3.10 -y
conda activate mgs

# Install PyTorch (adjust according to your CUDA version)
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt
```

### 2. Misc.

For Flash Attention. We recommend using the --no-build-isolation flag to prevent common compilation issues (refer to https://github.com/Dao-AILab/flash-attention for more installation help).

## 🛠️ Implementation Details

### Mixed Training Strategy
We provide a `MultiSourceRLHFDataset` to maintain data from heterogeneous sources. Data proportions are managed via a `WeightedRandomSampler`, allowing for precise control over the batch distribution:

- **Source Code:** `mgs/training/grpo/verl/utils/dataset/mixed_dataset.py`
- **Configuration Example:**

```python
# Customizing data proportions (e.g., in minimal_examples/test_mixed_dataset.py)
desired_proportions = {
    'math': 0.7,  # 70% math data in each batch
    'chat': 0.3   # 30% chat data in each batch
}
```
The full training code is implemented in `mgs/training/grpo/verl/trainer/main_ppo_mixed.py`.

### MGS Implementation

MGS is implemented by modifying the `update_policy` method of the `DataParallelPPOActor` class.
- Production Code: `mgs/training/grpo/verl/workers/dp_actor.py`
- Reference Example: A global version of gradient surgery (PCGrad) is available in `minimal_examples/test_pcgrad.py` for comparison.


### Reward Functions

Reward functions for the verifiable rewards, i.e., Math and IF are provided in:
- Math: `mgs/training/grpo/verl/utils/reward_score/hf_math_verify.py`
- IFEval: `mgs/training/grpo/verl/utils/reward_score/ifeval_reward.py`

## 🚀 Getting Started

### Training

To launch MGS training, execute:
```bash
bash scripts/train/mgs_grpo_qwen_zero_think.sh
```
You need adjust the configurations such as model path and dataset path accordingly.

### Evaluation

We utilize specialized evaluation suites to ensure accuracy:
- Math Eval: [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) suite.
- General Eval: [RLMT](https://github.com/princeton-pli/RLMT) suite.

## 📦 Model Checkpoints

Trained model checkpoints are withheld during the anonymous review period and will be released upon publication.

## ❤️ Acknowledgment

This project is built upon the excellent work of the open-source community. We specifically thank the developers of:

* [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) (Evaluation suite)
* [RLMT](https://github.com/princeton-pli/RLMT) (Core framework foundation)

Part of our implementation is directly adapted or extended from these repositories.
