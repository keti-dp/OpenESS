from typing import *
import torch
import torch.nn as nn
from .encoders import *
from .decoders import *


class EncDec(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        input_length: int,
        output_length: int,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name

        # Set an encoder
        if encoder_name == 'cnn':
            self.encoder = CNN_encoder(input_length)
        else:
            raise ValueError(f'Encoder {encoder_name} is not supported yet.')

        # Set a decoder
        if decoder_name == 'vv':
            self.decoder = vec2vec_decoder(output_length)
        else:
            raise ValueError(f'Decoder {decoder_name} is not supported yet.')

        # Initialize the weight tensors
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor, soh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (1) Encoder: drive -> SOH
        soh_pred = self.encoder(x).squeeze()
        # (2) Decoder: SOH -> R_i
        ri_pred = self.decoder(soh.unsqueeze(-1))
        return soh_pred, ri_pred
