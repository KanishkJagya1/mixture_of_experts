from datasets import load_dataset

def load_xlsum_languages(languages=["en", "hi", "pa"]):
    lang_code_map = {
        "en": "english",
        "hi": "hindi",
        "pa": "punjabi"
    }

    all_data = {
        "train": [],
        "validation": [],
        "test": []
    }

    for lang_code in languages:
        lang_name = lang_code_map[lang_code]
        try:
            dataset = load_dataset("GEM/xlsum", lang_name)
            for split in dataset:
                dataset[split] = dataset[split].add_column("lang", [lang_code] * len(dataset[split]))
                all_data[split].append(dataset[split])
        except Exception as e:
            print(f"Failed to load {lang_code}: {e}")

    return all_data