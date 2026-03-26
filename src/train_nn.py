import re
import cv2
import torch
import torch.nn as nn
from pathlib import Path


class SpeechDetectorNN(nn.Module):
    """
    Bi-directional GRU trained on labelled landmark sequences.
    """

    def __init__(self, feat_dim=160, hidden=64, num_layers=2, dropout=0.3):
        super.__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)



def run_training(file_path):
    pass


if __name__ == "__main__":
    file_path = Path("data/videos")
    run_training(file_path)
