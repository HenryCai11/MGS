from datasets import load_dataset
import random

# ====== 配置 ======
input_path = "./test_clean.parquet"          # 原始数据集
output_path = "./test_clean_tagged.parquet"  # 输出文件
tags = ["think", "thinking", "thought", "reason", "reasoning"]

# ====== 加载 ======
print(f"Loading dataset from {input_path} ...")
dataset = load_dataset('parquet', data_files=input_path)
train_ds = dataset['train']

# ====== 处理函数 ======
def replace_data_source(example):
    example["data_source"] = random.choice(tags)
    return example

# ====== 执行替换 ======
print("Replacing data_source uniformly ...")
train_ds = train_ds.map(replace_data_source)

# ====== 保存为 Parquet ======
print(f"Saving updated dataset to {output_path} ...")
train_ds.to_parquet(output_path)

print("✅ Done!")
print("Example after modification:")
print(train_ds[0])
