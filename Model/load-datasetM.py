import os
import re
import torchaudio
import requests
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def translate_text(text, token, url):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token
    }
    prompt = "Please translate the following text into Chinese, do not output any other unrelated things, do not explain your translation, please just output plain Chinese text which is purely the translation of the given text:\n\n"
    response = requests.post(
        url,
        headers=headers,
        json={
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "user", "content": prompt+text}
            ],
            "stream": False
        }
    )
    if response.status_code == 200:
        response = response.json()
        answer = response['choices'][0]['message']['content']
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        answer = re.sub(r'\(.*?\)', '', answer, flags=re.DOTALL)
        answer = re.sub(r'（.*?）', '', answer, flags=re.DOTALL)
        return answer.replace("\n", "")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def process_row(row, audio_path, token, url):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    text = " ".join(row.split(" ")[1:])
    chinese_translation = translate_text(text, token, url)
    return {
        'audio': {
            'path': audio_path,
            'array': audio_tensor.numpy().tolist(),
            'sampling_rate': sample_rate
        },
        'text': text,
        'chinese': chinese_translation
    }

def main():
    parser = argparse.ArgumentParser(description="Load dataset and process audio/text with multithreaded translation.")
    parser.add_argument('--fildir', type=str, required=True, help='Directory containing the audio files and text files.')
    parser.add_argument('--threads', type=int, default=24, help='Number of threads for translation API requests.')
    args = parser.parse_args()
    fildir = args.fildir
    num_threads = args.threads

    ollamaurl = "https://api.deepseek.com/chat/completions"
    token = open("Model/ollama.token", "r").read().replace("\n", "")

    data = []
    tasks = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for f in os.listdir(fildir):
            if f[0] == '.':
                continue
            for folder in os.listdir(os.path.join(fildir, f)):
                if folder[0] == '.':
                    continue
                trans_path = f"{fildir}/{f}/{folder}/{f}-{folder}.trans.txt"
                if not os.path.exists(trans_path):
                    continue
                with open(trans_path, "r", encoding="utf-8") as f1:
                    text = f1.read()
                text = text.split("\n")
                for row in text:
                    if row == "":
                        continue
                    audio_path = f"{fildir}/{f}/{folder}/{row.split(' ')[0]}.flac"
                    if not os.path.exists(audio_path):
                        continue
                    tasks.append(executor.submit(process_row, row, audio_path, token, ollamaurl))
        for future in as_completed(tasks):
            result = future.result()
            data.append(result)
            print(f"Processed {result['audio']['path']}, Chinese translation: {result['chinese']}")

    out_path = f"Model/data/{fildir.split('/')[-1]}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
