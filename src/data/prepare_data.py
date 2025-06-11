# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.loader import load_xlsum_languages
from src.data.preprocess import get_tokenizer, preprocess_dataset

#Load all 3 language datasets
datasets = load_xlsum_languages()
 
#Combine splits
from datasets import DatasetDict, concatenate_datasets

train_data, val_data, test_data = [], [], []
for lang_code in datasets:
    dset = datasets[lang_code]
    train_data.append(dset["train"])
    val_data.append(dset["validation"])
    test_data.append(dset["test"])

full_dataset = DatasetDict({
    "train": concatenate_datasets(train_data),
    "validation": concatenate_datasets(val_data),
    "test": concatenate_datasets(test_data),
})

#Tokenize
tokenizer = get_tokenizer("google/mt5-small")
tokenized_dataset = DatasetDict({
    split: preprocess_dataset(ds, tokenizer)
    for split, ds in full_dataset.items()
})

#Save (optional)
tokenized_dataset.save_to_disk("datasets/tokenized_xlsum")
from datasets import load_from_disk
ds = load_from_disk("datasets/tokenized_xlsum")
print(ds)
