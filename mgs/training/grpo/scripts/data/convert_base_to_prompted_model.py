#!/usr/bin/env python3
"""Create a "prompted" think-template variant of a base causal LM.

Adapted verbatim from
https://github.com/princeton-pli/RLMT/blob/main/training/sft_dpo/utils/src/convert_base_to_prompted_model.py
so MGS training on Qwen2.5-* base models can reuse the exact chat template
expected by ``main_ppo_mixed`` and the longcot reward pipeline.

The template instructs the model to think inside ``<think>...</think>`` and
answer inside ``<response>...</response>``. The script also writes
``stop.txt`` and ``longcot_config.json`` next to the converted checkpoint.

Usage:
  python convert_base_to_prompted_model.py \
      --model_path /path/to/Qwen2.5-14B \
      --out_path   /path/to/Qwen2.5-14B-Prompted \
      [--test_generation]
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


THINKING_CHAT_TEMPLATE = '''{{- bos_token }}
{{- "A conversation between User and Assistant. Following the User's query, the Assistant first plans a response, and then provides the response. " }}
{{- "The internal reasoning process is enclosed within <think> </think> tags and the response is enclosed within <response> </response> tags, " }}
{{- "i.e., in the format <think> reasoning process here </think> <response> response here </response>.\\n\\n" }}
{%- for message in messages %}
    {%- if message.role == 'user' %}
        {{- "User: <query>" + message['content'] | trim + "</query>\\n" }}
    {%- elif message.role == 'assistant' %}
        {{- "Assistant: " + message['content'] | trim }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- "Assistant:" }}
{%- endif %}'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert base model to prompted (with thinking) chat model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model to load")
    parser.add_argument("--out_path", type=str, required=True, help="Path to write converted model")
    parser.add_argument("--test_generation", action="store_true", help="Run a quick generation test (requires GPU or CPU)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser


def select_eos_token(model_path: str) -> str:
    lower = model_path.lower()
    if "llama" in lower:
        return "<|eot_id|>"
    return "<|im_end|>"


def maybe_test_generation(model, tokenizer, eos_token_id, max_new_tokens, temperature) -> None:
    messages = [{"role": "user", "content": "What is 2+2?"}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer.encode(formatted, return_tensors="pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    input_ids = input_ids.to(device)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(input_ids, generation_config=gen_cfg)
    generated_text = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=False)
    print("\n=== Sample generation ===")
    print(generated_text)


def main() -> None:
    args = build_parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")

    tokenizer.chat_template = THINKING_CHAT_TEMPLATE

    eos_token = select_eos_token(args.model_path)
    encoded = tokenizer.encode(eos_token, add_special_tokens=False)
    if not encoded:
        eos_token_id = tokenizer.eos_token_id
    else:
        eos_token_id = encoded[0]
        tokenizer.eos_token = eos_token
        tokenizer.eos_token_id = eos_token_id
    model.config.eos_token_id = eos_token_id

    out_path = Path(args.out_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.test_generation:
        try:
            maybe_test_generation(model, tokenizer, eos_token_id, args.max_new_tokens, args.temperature)
        except Exception as e:
            print(f"Warning: test generation failed: {e}")

    tokenizer.save_pretrained(str(out_path))
    model.save_pretrained(str(out_path))

    with open(out_path / "stop.txt", "w") as f:
        f.write("</response>")

    thinking_config = {
        "longcot": True,
        "longcot_delimiter": "<response>",
        "end_delimiter": None,
        "start_think_marker": "<think>",
    }
    with open(out_path / "longcot_config.json", "w") as f:
        json.dump(thinking_config, f)

    print(f"\nConverted model saved to {out_path}")


if __name__ == "__main__":
    main()
