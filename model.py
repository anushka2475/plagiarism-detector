import torch
import torch.nn as nn

class SiameseBiLSTM(nn.Module):
    def __init__(self, emb, hidden=128):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(emb.shape[1], hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2 * 2, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def encode(self, x):
        out, _ = self.lstm(self.emb(x))
        out = out.masked_fill((x == 0).unsqueeze(-1), -1e9)
        return out.max(dim=1).values

    def forward(self, a, b):
        va, vb = self.encode(a), self.encode(b)
        return self.fc(torch.cat([torch.abs(va - vb), va * vb], dim=1)).squeeze(1)