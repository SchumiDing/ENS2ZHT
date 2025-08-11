import torch
import torchaudio
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.en2zh import en2zh

import argparse
parser = argparse.ArgumentParser(description="Translate English Audio to Chinese Text.")
parser.add_argument('--audio_file', type=str, required=True, help='Path to the audio file to be translated')
parser.add_argument('--device', type=str, default='mps', help='Device to use for translation (default: mps)')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
args = parser.parse_args()

if __name__ == "__main__":
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

    model = en2zh().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    audio_input = torchaudio.load(args.audio_file)[0].to(device)
    audio_input = model.audioTransform(audio_input)

    audio_input = audio_input.unsqueeze(0)  # Add batch dimension
    
    output = model.autoRegressor(audio_input)
        
    
    translated_text = model.tokenizemodel.vector_to_token_ids(output[0].cpu().numpy())
    translated_text = model.tokenizemodel.decode_tokens(translated_text)
    
    print(f"Translated Text: {translated_text}")