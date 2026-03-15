from datasets import load_dataset, Dataset, DatasetDict
import argparse
from typing import Dict, Any, List

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{{}}"

def _build_prompt_from_question(q: str) -> List[Dict[str, str]]:
    """
    返回新的对话，仅保留一个 user 轮次：
    [{"role": "user", "content": "<question>\\nPlease reason ... \\boxed{{}}"}]
    """
    # 这里用双反斜杠，写入后就是单个 \boxed{ }（LaTeX）
    return [{"role": "user", "content": f"{q}\n{INSTRUCTION}"}]

def transform_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 example['prompt'] 改写为上述形式。
    优先从 extra_info['question'] 取题面；若无则回退到已存在的 prompt 内容做保底。
    """
    # 1) 拿 question（优先 extra_info.question）
    question = None
    if "extra_info" in example and isinstance(example["extra_info"], dict):
        question = example["extra_info"].get("question")

    # 2) 兜底：尝试从现有 prompt 中提取用户消息（若你的数据都含有 extra_info.question，可忽略这一段）
    if question is None:
        try:
            msgs = example.get("prompt", [])
            # 找 user 角色；没有就取第一条
            user_msg = next((m for m in msgs if m.get("role") == "user"), None) or (msgs[0] if msgs else None)
            question = user_msg.get("content") if user_msg else ""
        except Exception:
            question = ""

    # 3) 重写 prompt
    example["prompt"] = _build_prompt_from_question(question)
    return example

if __name__ == '__main__':
    ds = load_dataset("parquet", data_files="./train.parquet")
    updated = ds.map(transform_example, desc="Rewriting prompts -> question + instruction")

    updated['train'].to_parquet("./train_clean.parquet")

