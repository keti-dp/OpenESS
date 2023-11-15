import os, sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn

from submodels import FeatureAggregationLayer, select_last



# Simple RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, use_aggregation=False):
        super(SimpleRNN, self).__init__()        
        self.rnn= nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        if use_aggregation:
            self.aggregation_layer = FeatureAggregationLayer(hidden_size)
        else:
            self.aggregation_layer = select_last
        self.linear_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        x : (sliding_window, batch_size, feature_dim)
        """
        
        out, _ = self.rnn(x)
        out = out.transpose(0, 1).contiguous()

        last_out = self.aggregation_layer(out)
        return self.linear_layer(last_out)



# Simple LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, use_aggregation=False):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        if use_aggregation:
            self.aggregation_layer = FeatureAggregationLayer(hidden_size)
        else:
            self.aggregation_layer = select_last
        self.linear_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        x : (sliding_window, batch_size, feature_dim)
        """
        
        out, _ = self.lstm(x)
        out = out.transpose(0, 1).contiguous()

        last_out = self.aggregation_layer(out)
        return self.linear_layer(last_out)