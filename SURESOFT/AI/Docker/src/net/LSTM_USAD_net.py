import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super().__init__()
        if input_dim == 1:
            hidden_dim = 1
        else:
            hidden_dim = input_dim // 2
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm1(x)
        latent = self.hidden_to_latent(h[-1])

        return latent


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_layers):
        super().__init__()
        if output_dim == 1:
            hidden_dim = 1
        else:
            hidden_dim = output_dim // 2
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, seq_len):
        hidden = self.latent_to_hidden(x).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(hidden)
        output = self.output_layer(out)

        return output
