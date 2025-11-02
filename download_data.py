from datasets import load_dataset
import os

# 下载并保存数据集
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

# 可选：保存到本地
dataset.save_to_disk("wikitext-103-raw-v1")
print("Dataset downloaded and saved to 'wikitext-103-raw-v1/'")
