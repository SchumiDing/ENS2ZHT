from datasets import load_dataset

ds = load_dataset("MLCommons/peoples_speech", "clean")

ds.save_to_disk("Model/data/peoples_speech_clean")

data = []

for (i, example) in enumerate(ds['train']):
    data.append({
        'audio': example['audio'],
        'text': example['text']
    })

import json
with open("Model/data/peoples_speech_clean.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)