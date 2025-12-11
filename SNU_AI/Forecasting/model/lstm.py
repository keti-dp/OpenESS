import torch
import torch.nn as nn

class lstm_generator(nn.Module):
    """
    input:  (B, 86400, 7) TIMESTAMP 칼럼 제거
    output:  (B, 86400, 7)
    """
    def __init__(self, input_dim=7, hidden=256, layers=2, dropout=0.1, output_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, output_dim)

        # Initialize the weight tensors
        self.init_weights()

    
    def init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, h=None):
        # x: [B, T, C]
        y, h = self.lstm(x, h)     # y: [B, T, H]
        out = self.head(y)         # out: [B, T, output_dim]
        return out, h
