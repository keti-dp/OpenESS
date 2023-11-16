import torch
import torch.nn as nn


# <1d cnn> ########################################################################################

# input tensor size? (Batch_size, in_channels, Time_step)
class CNN_model(nn.Module):
    def __init__(self, num_col: int, input_length: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(num_col, 64, 5, 1, 2),  # nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
            nn.LeakyReLU(),
            # nn.AvgPool1d(2),            # nn.AvgPool1d(kernel_size)   /   nn.MaxPool1d(kernel_size)
            nn.Conv1d(64, 64, 5, 1, 2),
            nn.LeakyReLU(),
            # nn.AvgPool1d(2),
            nn.Conv1d(64, 64, 5, 1, 2),
            nn.LeakyReLU(),
            # nn.AvgPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_length * 64, 128), # nn.Linear(in_features, out_features)
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x) #.transpose(1, 2))  # (batch_size, hidden_dim, input_dim) --> (batch_size, input_dim, hidden_dim)
        out = out.view(out.shape[0], -1)    # view(contig), transpose(not contig)
        out = self.fc(out)
        return out
        
        

# <ResNet 18, 34, 50 101, 152> #################################################################

class BasicBlock(nn.Module):
    # For resnet 18 and resnet 34

    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion)
        )

        # identity mapping
        self.shortcut = nn.Sequential()

        # he shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    # For resnet over 50 layers
    
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels * BottleNeck.expansion),
        )
        
        # identity mapping
        self.shortcut = nn.Sequential()

        # to match the dimension
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_col):
        super().__init__()

        self.in_channels = 64 
        
        # (B, 3, T) => (B, 64, T)
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_col, 64, kernel_size=3, padding=1, bias=False), # <-------------------------in_channel = 3
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, 1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        block: block type, basic block or bottle neck block
        out_channels: output depth channel number of this layer
        num_blocks: how many blocks per layer
        stride: the stride of the first block of this layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):       # raw x shape (B, C, T) = (B, 3, T)
        output = self.conv1(x)  # (B, C, T) = (B, 64, T)
        output = self.conv2_x(output)   # (B, C, T) = (B, 64, T)
        output = self.conv3_x(output)   # (B, C, T) = (B, 128, T)
        output = self.conv4_x(output)   # (B, C, T) = (B, 256, T)
        output = self.conv5_x(output)   # (B, C, T) = (B, 512, T)
        output = self.avg_pool(output)  # (B, C, T) = (B, 512, 1)
        output = output.view(output.size(0), -1) # (B, 512)
        output = self.fc(output)

        return output
    

def resnet10(num_col):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_col)    

def resnet18(num_col):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_col)

def resnet34(num_col):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_col)

def resnet50(num_col):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_col)

def resnet101(num_col):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_col)

def resnet152(num_col):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_col)



# <LSTM> ############################################################################

class Conv1d_LSTM(nn.Module):
    def __init__(self, num_col, out_channel=1):
        super(Conv1d_LSTM, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=num_col, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=50, num_layers=1, bias=True, bidirectional=False, batch_first=True)
        
        self.dropout = nn.Dropout(0.5)

        self.dense1 = nn.Linear(50, 32)
        self.dense2 = nn.Linear(32, out_channel)

    def forward(self, x):   # Raw x shape : (B, S, F) => (B, 10, 3) batch_size, sequence, feature
        x = x.transpose(1, 2) # Shape : (B, F, S) => (B, 3, 10)
        x = self.conv1d_1(x) # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 16, 10)
        x = self.conv1d_2(x) # Shape : (B, C, S) => (B, 32, 10)
        x = x.transpose(1, 2) # Shape : (B, S, C) == (B, S, F) => (B, 10, 32)
        
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x) # Shape : (B, S, H) // H = hidden_size => (B, 10, 50)
        x = hidden[-1] # Shape : (B, H) // -1 means the last sequence => (B, 50)
    
        x = self.dropout(x) # Shape : (B, H) => (B, 50)
        
        x = self.fc_layer1(x) # Shape : (B, 32)
        x = self.fc_layer2(x) # Shape : (B, O) // O = output => (B, 1)

        return x


def CustomModel(config):
    num_col = len(config['input_cols'])
    if config['model'] == 'cnn':
        return CNN_model(num_col=num_col, input_length=config['seconds'])
    elif config['model'] == 'res10':
        return resnet10(num_col)
    elif config['model'] == 'res18':
        return resnet18(num_col)
    elif config['model'] == 'res34':
        return resnet34(num_col)
    elif config['model'] == 'res50':
        return resnet50(num_col)
    elif config['model'] == 'res101':
        return resnet101(num_col)
    elif config['model'] == 'res152':
        return resnet152(num_col)
    elif config['model'] == 'LSTM':
        return Conv1d_LSTM(num_col)