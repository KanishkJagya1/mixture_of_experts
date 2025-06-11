from transformers import AutoTokenizer

def get_tokenizer(model_name="google/mt5-small", use_fast=False):
    return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

def preprocess_dataset(dataset, tokenizer, max_input_len=512, max_target_len=64):
    def tokenize_fn(example):
        model_input = tokenizer(
            example["text"],
            max_length=max_input_len,
            padding="max_length",
            truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["summary"],
                max_length=max_target_len,
                padding="max_length",
                truncation=True
            )
        model_input["labels"] = labels["input_ids"]
        model_input["lang"] = example["lang"]
        return model_input

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
