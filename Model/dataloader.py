from torch.utils.data import Dataset
import torch

class audioTextDataset(Dataset):
    def __init__(self, Audios, tgts, targets,interval=16000, device="mps"):
        self.Audios = Audios
        self.tgts = tgts
        self.targets = targets
        self.interval = interval
        self.device = device
        
    def __len__(self):
        return len(self.Audios)

    def __getitem__(self, idx):
        audio = self.Audios[idx]
        tgt = self.tgts[idx]
        target = self.targets[idx]
        audioTensor = torch.load(audio, map_location=self.device)
        tgtTensor = torch.load(tgt, map_location=self.device)
        targetTensor = torch.load(target, map_location=self.device)
        return audioTensor, tgtTensor, targetTensor