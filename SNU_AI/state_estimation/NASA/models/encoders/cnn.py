import torch
import torch.nn as nn


class CNN_encoder(nn.Module):
    def __init__(self, input_length: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(2, 64, 5, 1, 2),
            nn.LeakyReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(64, 64, 5, 1, 2),
            nn.LeakyReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(64, 64, 5, 1, 2),
            nn.LeakyReLU(),
            nn.AvgPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_length // 8 * 64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x.transpose(1, 2))
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
        

