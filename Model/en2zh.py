import torch, torchaudio
import sys
import os

# Device configuration
device = "cpu"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .tokenizer import ChineseBertTokenizer
except ImportError:
    from tokenizer import ChineseBertTokenizer

class en2zh(torch.nn.Module):
    def __init__(self):
        super(en2zh, self).__init__()
        self.interval = 768
        self.step = 32
        self.output = 768
        self.tokenizemodel = ChineseBertTokenizer("hfl/chinese-bert-wwm-ext")
        self.final = torch.nn.Sequential(
            torch.nn.Linear(self.interval, self.interval),
            torch.nn.ReLU(),
            torch.nn.Linear(self.interval, self.interval),
            torch.nn.ReLU(),
            torch.nn.Linear(self.interval, self.interval),
            torch.nn.ReLU(),
            torch.nn.Linear(self.interval, self.interval),
            torch.nn.ReLU(),
            torch.nn.Linear(self.interval, self.output,)
        )
        self.transformers = torch.nn.Sequential(
            torch.nn.TransformerEncoderLayer(
                d_model=self.interval,
                nhead=8,
                dim_feedforward=2048,
                dropout=0,
                activation='relu'
            ),
            torch.nn.TransformerEncoderLayer(
                d_model=self.interval,
                nhead=8,
                dim_feedforward=2048,
                dropout=0,
                activation='relu'
            ),
            torch.nn.TransformerEncoderLayer(
                d_model=self.interval,
                nhead=8,
                dim_feedforward=2048,
                dropout=0,
                activation='relu'
            )
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.to(device)
        
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
        

    def audioTransform(self, audio: torch.Tensor):

        output = torch.empty((audio.shape[-1]- self.interval)//self.step, self.interval).to(device)
        for i in range(output.shape[0]):
            start = i * self.step
            end = start + self.interval
            segment = audio[0][start:end]
            output[i] = segment
        return output
    
    def forward(self, audio: torch.Tensor):
        audio = self.transformers(audio)
        audio = self.final(audio)
        return audio
        
    def autoRegressor(self, inputAudio:torch.tensor):
        
        inputAudio = inputAudio.to(device)
        while True:
            output = self.forward(inputAudio)
            newtoken = output[-1].unsqueeze(0)
            inputAudio = torch.cat((inputAudio, newtoken), dim=0)
            newid = self.tokenizemodel.vector_to_token_ids(newtoken, top_k=1)
            if newid[0][-1] == self.tokenizemodel.tokenizer.sep_token_id:
                break
            yield self.tokenizemodel.decode_tokens(newid[0][-1])
    
    def autoRegressorTraining(self, inputAudio:torch.tensor, targetText:str, epoches=1, log=False):
        
        inputAudio = inputAudio.to(device)
        targetTokens = self.tokenizemodel.to_vector(targetText)['last_hidden_state']
        targetTokens = targetTokens.to(device)
        l = len(targetTokens[0])
        lA = len(inputAudio)
        loss = 0
        for _ in range(epoches):
            loss = 0
            for i in range(l):
                output = self.forward(inputAudio)
                newtoken = output[-1].unsqueeze(0)
                loss = self.criterion(newtoken[0], targetTokens[0][i])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                inputAudio = torch.cat((inputAudio, newtoken.clone().detach()), dim=0)
                if log:
                    print(f"Step {i+1}/{l}, Loss: {loss.item()}")
                loss += loss.item()
            loss = loss / l

        torch.save(self.state_dict(), 'Model/pth/en2zh_model.pth')
        return loss
        

if __name__ == "__main__":
    model = en2zh()
    
    # Example usage
    audio_input = torch.randn(1, 768*1000).to(device)  # Dummy audio input
    audio_transformed = model.audioTransform(audio_input)
    print("Transformed Audio Shape:", audio_transformed.shape)
    
    target_text = "我喜欢自然语言处理。"
    model.autoRegressorTraining(audio_transformed, target_text, epoches=5)
    