import numpy as np
import torch
import torch.nn as nn



# Select the last component of RNN output.
def select_last(x):
    last_component = torch.zeros(x.shape[1], device=x.device)
    last_component[-1] = 1
    return torch.matmul(last_component, x)



# Layer aggregating features using self-attention algorithm
class FeatureAggregationLayer(nn.Module):
    def __init__(self, d_embed, dropout=0.1):
        super(FeatureAggregationLayer, self).__init__()
        self.linear_layer = nn.Linear(d_embed, 1)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.scale = 1 / np.sqrt(d_embed)
        
        nn.init.xavier_uniform_(self.linear_layer.weight)
        
    def forward(self, x):
        """
        x : (n_batch, seq_len, d_embed)
        """
        scores = self.linear_layer(x).squeeze(dim=-1) * self.scale
        weights = self.softmax_layer(scores)
        return self.dropout_layer(torch.matmul(weights.unsqueeze(1), x).squeeze(1))