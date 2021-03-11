import torch
import torch.nn as nn


class SimpleDenseNet(nn.Module):
    def __init__(self, input_dims: int, out_dims: int):
        super().__init__()
        self.seq_dense = nn.Sequential(
            nn.Linear(input_dims, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, out_dims),
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        out = self.seq_dense(state)
        return out
