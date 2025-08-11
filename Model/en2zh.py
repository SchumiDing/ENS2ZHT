import torch, torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn.functional as F
import sys
import os

# Device configuration
device = "cpu"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .tokenizer import ChineseBertTokenizer
except ImportError:
    from tokenizer import ChineseBertTokenizer

from Model.dataloader import audioTextDataset

class en2zh(torch.nn.Module):
    def __init__(self):
        super(en2zh, self).__init__()
        self.interval = 768
        self.step = 512
        self.output = 768
        self.tokenizemodel = ChineseBertTokenizer()
        # Initialize Wav2Vec2 for feature extraction
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.final = torch.nn.Sequential(
            torch.nn.Linear(self.interval, self.interval),
            torch.nn.ReLU(),
            torch.nn.Linear(self.interval, self.interval),
            torch.nn.ReLU(),
            torch.nn.Linear(self.interval, self.output)
        )
        self.transformers = torch.nn.Sequential(
            torch.nn.Transformer(
                d_model=self.interval,
                nhead=8,
                num_encoder_layers=12,
                num_decoder_layers=12,
                dim_feedforward=2048,
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
        )
        self.criterion = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.to(device)
        # self.project = torch.nn.Linear(768, 10000, bias=False)
        
        for param in self.transformers.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
        for param in self.final.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
        
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
        

    def audioTransform(self, audio: torch.Tensor):
        # Use Wav2Vec2 to extract deep features and pad to fixed size
        # audio shape: (1, seq_length)
        input_values = self.processor(audio.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values.to(audio.device)
        outputs = self.wav2vec2(input_values)
        features = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_size=768)
        # Pad or truncate to (1250, 768)
        max_len = 1250
        seq_len = features.size(0)
        if seq_len < max_len:
            features = F.pad(features, (0, 0, 0, max_len - seq_len))
        else:
            features = features[:max_len]
        print(f"[en2zh.py] Padded features shape: {features.shape}")
        return features
    
    def forward(self, audio: torch.Tensor, tgt=None):
        # 如果tgt为None，则用audio自身作为tgt（仅用于演示，实际可根据任务调整）
        if tgt is None:
            tgt = audio
        for trans in self.transformers:
            # 只对Transformer实例传src和tgt
            audio = trans(audio, tgt)
        audio = self.final(audio)
        return audio
        
    def autoRegressor(self, inputAudio:torch.tensor):
        inputAudio = inputAudio.to(device)
        while True:
            output = self.forward(inputAudio, inputAudio)
            newtoken = output[-1].unsqueeze(0)
            inputAudio = torch.cat((inputAudio, newtoken), dim=0)
            newid = self.tokenizemodel.vector_to_token_ids(newtoken, top_k=1)
            if newid[0][-1] == self.tokenizemodel.tokenizer.sep_token_id:
                break
            yield self.tokenizemodel.decode_tokens(newid[0][-1])
    
    def autoRegressorTraining(self, inputAudio:torch.tensor, targetText:str, epoches=1, log=True):
        inputAudio = inputAudio.to(device)
        targetTokens = self.tokenizemodel.to_vector(targetText)['last_hidden_state']
        targetTokens = targetTokens.to(device)
        l = len(targetTokens[0])
        lA = len(inputAudio)
        loss = 0
        outAudio = torch.empty((0, self.interval)).to(device)
        for _ in range(epoches):
            loss = 0
            for i in range(l):
                if i == 0:
                    outAudio = targetTokens[0][0].unsqueeze(0)
                else:
                    newtoken = output[-1].unsqueeze(0)
                    outAudio = torch.cat((outAudio, newtoken.clone().detach()), dim=0)
                output = self.forward(torch.cat((inputAudio, outAudio), dim=0), outAudio)
                loss = self.criterion(newtoken[0], targetTokens[0][i])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if log:
                    print(f"Step {i+1}/{l}, Loss: {loss.item()}")
                loss += loss.item()
            loss = loss / l

        torch.save(self.state_dict(), 'Model/pth/en2zh_model.pth')
        return loss
    
    def createBatchTrainData(self, Audios: list[torch.Tensor], targetTexts: list[str], batch_size=32, device="mps"):
        batchAudio = []
        batchTarget = []
        aims = []
        traindata = []
        
        cnt = 0
        
        for audio, text in zip(Audios, targetTexts):
            audioTensor = torch.load(audio, map_location=device)
            audioTensor = audioTensor.cpu()
            audioTensor = self.audioTransform(audioTensor)
            audioTensor = audioTensor.to(device)
            d = self.tokenizemodel.to_vector(text)
            targetTokens = d['last_hidden_state'].to(device)
            stop_token_length = d['stop_token_length']
            outAudio = torch.empty((len(targetTokens[0]), self.interval)).to(device)
            
            torch.save(audioTensor, f"Model/data/audio_{cnt}.pt")
            if not os.path.exists(f"Model/data/audio_{cnt}"):
                os.makedirs(f"Model/data/audio_{cnt}")
            for i in range(len(targetTokens[0])):
                if stop_token_length <= 0:
                    break
                if i == 1:
                    outAudio[0] = targetTokens[0][0]
                elif i > 1:
                    outAudio[i-1] = targetTokens[0][i]
                
                torch.save(targetTokens[0][i].unsqueeze(0), f"Model/data/audio_{cnt}/target_{i}.pt")
                torch.save(outAudio, f"Model/data/audio_{cnt}/outAudio_{i}.pt")
                
                aims.append(f"Model/data/audio_{cnt}/target_{i}.pt")
                batchAudio.append(f"Model/data/audio_{cnt}.pt")
                batchTarget.append(f"Model/data/audio_{cnt}/outAudio_{i}.pt")
                stop_token_length -= 1
            cnt += 1
        self.traindata = audioTextDataset(batchAudio, batchTarget, aims, device=device)
        print(f"[en2zh.py] Created {len(self.traindata)} training samples.")
        self.dataloader = torch.utils.data.DataLoader(self.traindata, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"[en2zh.py] Created DataLoader with batch size {batch_size}.")

        return self.dataloader

if __name__ == "__main__":
    model = en2zh()
    
    # Example usage
    audio_input = torch.randn(1, 768*1000).to(device)  # Dummy audio input
    audio_transformed = model.audioTransform(audio_input)
    print("[en2zh.py] Transformed Audio Shape:", audio_transformed.shape)

    target_text = "我喜欢自然语言处理。"
    model.autoRegressorTraining(audio_transformed, target_text, epoches=5)
