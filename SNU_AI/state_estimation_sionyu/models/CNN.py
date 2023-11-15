import numpy as np
import pandas as pd
import torch.nn as nn
import torch

class NN1(nn.Module) :  ## OCV targeting
    def __init__ (self, input_length : int ,num_col = 4) : ## num_col: V,I,T,Vgap 채널의갯수
        super().__init__()
        self.input_length = input_length
        self.main = nn.Sequential(
            nn.Conv1d(num_col,64,5,2,2),  ## output_size = input size/2
            nn.LeakyReLU(),
            nn.Conv1d(64,128,5,2,2),      ## output_size = input size/2
            nn.LeakyReLU(),
            nn.Conv1d(128,256,5,2,2), 
            nn.LeakyReLU(),
            nn.Conv1d(256,512,5,2,2),
            nn.LeakyReLU(),
        )
        
        self.output_length = (self.input_length-1)//16 + 1
        
        self.fc = nn.Sequential(                                        ##fc layer 때문에 time series 의 정보 다 잃지않을까...?
            nn.Linear(self.output_length * 512 , 1024),
            nn.LeakyReLU(),
            nn.Linear(1024,self.input_length),
        )
        
        
    def forward(self,x: torch.Tensor) -> torch.Tensor : ## x : V,I,T,Vgap  time series data
        out = self.main(x)
        out = out.view(-1,self.output_length * 512)
        out = self.fc(out)
        return out
    
    # def train(self,x : torch.Tensor, y : torch.Tensor) : ## x : time series data , y : OCV GT
    #     z = predict(self,x)
    #     loss = torch.sum((z-y)**2)/self.input_length     ## 사용하는 loss가 MSE loss가 맞나...?
    #     loss.backward()
        

        
class NN2(nn.Module) : ## SOH targeting (with mask if no OCV data)
    def __init__(self ,  input_length : int ,num_col = 5, mask = None) : ##num_col: V,I,T,Vgap, (OCV 없으면 +1 해서 입력) 채널의 갯수 // OCV column의 존재 여부 boolean
        super().__init__() 
        self.input_length = input_length
        self.mask = mask
        self.main = nn.Sequential(
            nn.Conv1d(num_col,64,5,2,2),  ## output_size = input size/2
            nn.LeakyReLU(),
            nn.Conv1d(64,128,5,2,2),      ## output_size = input size/2
            nn.LeakyReLU(),
            nn.Conv1d(128,256,5,2,2), 
            nn.LeakyReLU(),
            nn.Conv1d(256,512,5,2,2),
            nn.LeakyReLU(),
        )
        
        self.output_length = (self.input_length-1)//16 + 1  ##input의 길이가 16이면 1, 17~32이면 2 이런식으로 나와야함
        
        self.fc = nn.Sequential(
            nn.Linear(self.output_length * 512 , 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )
        
        
        
    def forward(self,x : torch.Tensor) -> torch.Tensor : ## x : time series data (dataframe 형태로 들어온다면 ? (no. tensor형태))
        # if 'OCV' in x.columns :
        #     self.OCV = True
        # x : N x num_col x input_length
        if self.mask:
            masked_column = torch.zeros(x.shape[0], 1, self.input_length, device=x.device)  # input_length x 1
            # x = x.view(x.shape[0]*self.input_length,-1)    # num_col * input_length x 1
            x = torch.concat([x, masked_column], dim=1)                  # (num_col * input_length + input_length) x 1
            # x = x.view(-1,self.input_length)               # num_col(OCV포함) x input_length
            out = self.main(x)
        else :
            out = self.main(x)
        
        out = out.view(-1, self.output_length * 512)
        out = self.fc(out)
        
        return out
    
    # def train(self,x :torch.Tensor , SOH : torch.Tensor) : ## x: time series data , SOH(GT)
    #     z = predict(self,x)
    #     loss = torch.sum((z-SOH)**2)/self.input_length
    #     loss.backward()
            
            
        
        # self.df = pd.read_parquet('data')
        # self.V = self.df['V']
        # self.I = self.df['I']
        # self.T = self.df['T']
        # self.Vgap = self.df['VOLT_gap']
        # self.OCV = self.df['OCV_est']
        # self.SOC = self.df['SOC']
        # self.length = len(self.df)
        
        # self.conv1 = nn.Conv1d(4,64,5,1,2)