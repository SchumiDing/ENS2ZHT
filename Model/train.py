import torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model.en2zh import en2zh

import argparse
parser = argparse.ArgumentParser(description="Load dataset and train the model.")
parser.add_argument('--dataset', type=str, required=True, help='Directory of a json file with audio and text info')
args = parser.parse_args()
fil = args.dataset

device = "cpu"
parser.add_argument('--device', type=str, default='mps', help='Device to use for training (default: mps)')
args = parser.parse_args()
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

epoches = 10000
batch_size = 32
parser.add_argument('--epoches', type=int, default=10000, help='Number of training epochs (default: 10000)')
args = parser.parse_args()
if args.epoches > 10000:
    print("Warning: Epoches is set to a very high value, this may take a long time to train.")
elif args.epoches < 1:
    print("Warning: Epoches is set to a very low value, this may not train the model properly.")
else:
    epoches = args.epoches

import json, re
import numpy as np

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    model = en2zh().to(device)
    data = json.load(open(fil, "r"))
    
    training = []
    audio = []
    text = []
    for item in data:
        audio_tensor = torch.tensor(item['audio']['array'], dtype=torch.float32).to(device)
        audio_tensor = model.audioTransform(audio_tensor)
        # training.append({
        #     'audio': audio_tensor.to(device),
        #     'chinese': re.sub(r'\(.*?\)', '', item['chinese'], flags=re.DOTALL)
        # })
        audio.append(audio_tensor)
        text.append(item['text'])
    np.random.shuffle(training)
    train_data = model.createBatchTrainData(audio, text, batch_size=batch_size, device=device)

    print(f"Training data created with {len(train_data)} batches., each")
    for batch in train_data:
        audio_batch, text_batch = batch
        print(f" - Audio batch shape: {audio_batch.shape}, Text batch shape: {text_batch.shape}")
        break
    if not os.path.exists('Model/pth'):
        os.makedirs('Model/pth')
    for epoch in range(epoches):
        total_loss = 0
        for batch in train_data:
            audio_batch, text_batch = batch
            loss = model.autoRegressorTraining(audio_batch, text_batch, epoches=1, log=True)
            total_loss += loss
        
        print(f"Epoch {epoch+1}/{epoches}, Total Loss: {total_loss/len(train_data)}")
        torch.save(model.state_dict(), 'Model/pth/en2zh_model.pth')