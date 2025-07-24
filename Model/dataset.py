from datasets import load_dataset
import os
os.environ['HF_ENDPOINT'] = 'https://www.modelscope.cn'
# 可选：设置缓存目录
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'
ds = load_dataset("MLCommons/peoples_speech", "clean")

ds.save_to_disk("Model/data/peoples_speech_clean")

data = []

ollamaurl = "http://localhost:11434"

import requests

def translate_text(text):
    response = requests.post(
        ollamaurl + "/chat",
        json={
            "model": "deepseek:70b",
            "messages": [
                {"system": "You are a helpful assistant that translates English text to Chinese. Please only output the translated Chinese sentences based on sentences provided by the user and do not add any additional text."},
                {"role": "user", "content": text}
            ]
        }
    )
    if response.status_code == 200:
        return response.json()['message']['content']
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

for (i, example) in enumerate(ds['train']):
    data.append({
        'audio': example['audio'],
        'text': example['text'],
        'trans': translate_text(example['text'])
    })

import json
with open("Model/data/peoples_speech_clean.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)