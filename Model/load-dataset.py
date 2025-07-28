import os
import torchaudio

fildir = "Model/test-clean"

import requests
ollamaurl = "http://localhost:11434"
def translate_text(text):
    response = requests.post(
        ollamaurl + "/chat",
        json={
            "model": "deepseek-r1:70b",
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

data = []
for f in os.listdir(fildir):
    if f[0]=='.':
        continue
    for folder in os.listdir(os.path.join(fildir, f)):
        if folder[0]=='.':
            continue
        text = ""
        with open(f"{fildir}/{f}/{folder}/{f}-{folder}.trans.txt", "r", encoding="utf-8") as f1:
            text = f1.read()
        text = text.split("\n")
        for row in text:
            if row == "":
                continue
            audio_path = f"{fildir}/{f}/{folder}/{row.split(' ')[0]}.flac"
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            data.append({
                'audio': {
                    'path': audio_path,
                    'array': audio_tensor.numpy().tolist(),
                    'sampling_rate': sample_rate
                },
                'text': " ".join(row.split(" ")[1:]),
            })
            chinese_translation = translate_text(" ".join(row.split(" ")[1:]))
            data[-1]['chinese'] = chinese_translation

import json

json.dump(data, open(f"Model/data/{fildir.split('/')[-1]}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)