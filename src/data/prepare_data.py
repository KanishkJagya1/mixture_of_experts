from src.data.loader import load_xlsum_languages
from src.data.preprocess import get_tokenizer, preprocess_dataset
from datasets import DatasetDict, concatenate_datasets, load_from_disk

# Load datasets
datasets = load_xlsum_languages()

# Merge per split
full_dataset = DatasetDict({
    "train": concatenate_datasets(datasets["train"]),
    "validation": concatenate_datasets(datasets["validation"]),
    "test": concatenate_datasets(datasets["test"]),
})

# Use CPU-compatible tokenizer
tokenizer = get_tokenizer("google/mt5-small", use_fast=False)

# Preprocess
tokenized_dataset = DatasetDict({
    split: preprocess_dataset(ds, tokenizer)
    for split, ds in full_dataset.items()
})

# Save to disk
tokenized_dataset.save_to_disk("datasets/tokenized_xlsum")

# Verify
ds = load_from_disk("datasets/tokenized_xlsum")
print(ds)
