import torch
import torch.nn as nn


class vec2vec_decoder(nn.Module):
    def __init__(self, output_length: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, output_length),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.main(z)
        return out
        

