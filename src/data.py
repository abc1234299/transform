from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk

def get_tokenizer_and_data(data_dir="/root/data/wikitext-103-raw-v1", tokenizer_path="/root/autodl-tmp/data/gpt2_local"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(data_dir)
    train_data = dataset["train"].select(range(10000))
    val_data = dataset["validation"].select(range(1000))

    return tokenizer, train_data, val_data

def collate_fn(tokenizer, max_length=128):
    def collate(batch):
        texts = [item["text"] for item in batch]
        encoded = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        input_ids = encoded["input_ids"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        return {"input_ids": input_ids, "labels": labels}
    return collate