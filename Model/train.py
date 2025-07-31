import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model.en2zh import en2zh

import argparse
parser = argparse.ArgumentParser(description="Load dataset and train the model.")
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
parser.add_argument('--epoches', type=int, default=10000, help='Number of training epochs (default: 10000)')
parser.add_argument('--dataset', type=str, required=True, help='Directory of a json file with audio and text info')
parser.add_argument('--device', type=str, default='mps', help='Device to use for training (default: mps)')
parser.add_argument('--traindevice', type=str, default='mps', help='Model to use for training (default: en2zh)')
args = parser.parse_args()
fil = args.dataset

traindevice = args.traindevice

device = "cpu"
if args.device == "cpu":
    device = "cpu"
elif args.device == "mps":
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        print("MPS is not available, falling back to CPU.")
        device = "cpu"
elif args.device == "cuda":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA is not available, falling back to CPU.")
        device = "cpu"
elif args.device == "xpu":
    if torch.xpu.is_available():
        device = "xpu"
    else:
        print("XPU is not available, falling back to CPU.")
        device = "cpu"
else:
    print(f"Unknown device {args.device}, falling back to CPU.")
    device = "cpu"
print()
epoches = 10000
if args.epoches > 10000:
    print("Warning: Epoches is set to a very high value, this may take a long time to train.")
elif args.epoches < 1:
    print("Warning: Epoches is set to a very low value, this may not train the model properly.")
else:
    epoches = args.epoches

batch_size = 32

if args.batch_size < 1:
    raise ValueError("Batch size must be at least 1.")
elif args.batch_size > 256:
    batch_size = args.batch_size
    print("Warning: Batch size is set to a very high value, this may cause memory issues.")
else:
    batch_size = args.batch_size

import json, re
import numpy as np
import ijson

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    training = []
    audio = []
    text = []
    skipped = 0
    cnt = 0
    if not os.path.exists('Model/data/_temp'):
        os.makedirs('Model/data/_temp')
    print(f"[train.py] Start to initiate the model")
    model = en2zh().to(traindevice)
    print("[train.py] about to load tokenizer")
    with open(f"Model/data/{fil}.json", "r") as f:
        for item in ijson.items(f, 'item'):
            audio_tensor = item.get('audio').get('array')
            t = item.get('text')
            audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
            torch.save(audio_tensor, f"Model/data/_temp/audio_{cnt}.pt")
            audio.append(f"Model/data/_temp/audio_{cnt}.pt")
            text.append(t)
            cnt += 1
    print(f"[train.py] Loaded {len(audio)} audio samples and {len(text)} text samples from {fil}.json.")
    if skipped > 0:
        print(f"[train.py] Skipped {skipped} samples that had empty Chinese translations; using {len(audio)} valid samples.")
    
    train_data = model.createBatchTrainData(audio, text, batch_size=batch_size, device=device)

    print(f"[train.py] Training data created with {len(train_data)} batches.")
    if not os.path.exists('Model/pth'):
        os.makedirs('Model/pth')
    for epoch in range(epoches):
        total_loss = 0
        for batch in train_data:
            model.optimizer.zero_grad()
            audio_batch, tpt,  text_batch = batch
            audio_batch = audio_batch.to(traindevice)
            tpt = tpt.to(traindevice)
            text_batch = text_batch.to(traindevice)
            # print(audio_batch.shape, tpt.shape, text_batch.shape)
            ans = model.forward(audio_batch, tpt)
            loss = model.criterion(ans, text_batch)

            loss.backward()
            model.optimizer.step()
            total_loss += loss.item()
            print(f"[train.py] Epoch {epoch+1}/{epoches}, Batch Loss: {loss.item()}")
        print(f"[train.py] Epoch {epoch+1}/{epoches}, Avg Total Loss: {total_loss/len(train_data)}")
        torch.save(model.state_dict(), 'Model/pth/en2zh_model.pth')