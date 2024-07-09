import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features=1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))