import torch.nn as nn
class DeepAntNet(nn.Module):
    def __init__(self, n_feature):
        self.n_feature = n_feature
        self.seq_len = 10
        super().__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=n_feature, out_channels=32, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.flatten = nn.Flatten()
        
        self.denseblock = nn.Sequential(
            nn.Linear(32, 40),
            #nn.Linear(96, 40), # for SEQL_LEN = 20
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )
        self.out = nn.Linear(40, self.n_feature)
        
    def forward(self, x):
        
        x = self.convblock1(x)
        
        x = self.convblock2(x)
        x = self.flatten(x)
        x = self.denseblock(x)
        x = self.out(x)
        
        return x.view(-1, self.n_feature, 1)
    
    
    
    