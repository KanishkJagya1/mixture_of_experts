from datasets import load_dataset

def load_xlsum_languages(languages=["en", "hi", "pa"]):
    lang_code_map = {
        "en": "english",
        "hi": "hindi",
        "pa": "punjabi"
    }
    
    datasets = {}
    for lang_code in languages:
        lang_name = lang_code_map[lang_code]
        ds = load_dataset("GEM/xlsum", lang_name, trust_remote_code=True) 
        for split in ds:
            ds[split] = ds[split].add_column("lang", [lang_code] * len(ds[split]))
        datasets[lang_code] = ds

    return datasets