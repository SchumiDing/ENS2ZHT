import os, re
import torchaudio

fildir = "Model/test-clean"

import requests
ollamaurl = "http://127.0.0.1:11434/api/chat"
def translate_text(text):
    headers = {"Content-Type": "application/json"}
    prompt = "Please translate the following text into Chinese, do not output any other unrelated things:\n\n"
    response = requests.post(
        ollamaurl,
        headers=headers,
        json={
            "model": "deepseek-r1:70b",
            "messages": [
                {"role": "user", "content": prompt+text}
            ],
            "stream":False
        }
    )
    if response.status_code == 200:
        answer = response.json()['message']['content']
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        return answer.replace("\n", "")
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