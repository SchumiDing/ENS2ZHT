fildir = "Model/test-clean"

import argparse
import os, re
import torchaudio

parser = argparse.ArgumentParser(description="Load dataset and process audio/text.")
parser.add_argument('--fildir', type=str, required=True, help='Directory containing the audio files and text files.')
args = parser.parse_args()
fildir = args.fildir

import requests
ollamaurl = "https://api.deepseek.com/chat/completions"
def translate_text(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + open("Model/ollama.token", "r").read().replace("\n", "")
    }
    prompt = "Please translate the following text into Chinese, do not output any other unrelated things:\n\n"
    response = requests.post(
        ollamaurl,
        headers=headers,
        json={
            "model": "deepseek-reasoner",
            
            "messages": [
                {"role": "user", "content": prompt+text}
            ],
            "stream":False
        }
    )
    if response.status_code == 200:
        response = response.json()
        answer = response['choices'][0]['message']['content']
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
            print(f"Processed {audio_path}, Chinese translation: {chinese_translation}")

import json

json.dump(data, open(f"Model/data/{fildir.split('/')[-1]}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)