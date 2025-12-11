import os
import math
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from soc_slope_synthesis import calculate_soc_slope
from npy_to_parquet import npy_to_parquet

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Supports variable sequence lengths up to max_len.
    """
    def __init__(self, d_model: int, max_len: int = 100_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        Returns:
            x + positional encoding: same shape
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].unsqueeze(1)


class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based encoder for multivariate time series.
    Input:  [batch_size, seq_len, input_dim]
    Output: [batch_size, output_dim]
    """
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_dim: int = 10,
        dropout: float = 0.1,
        max_len: int = 100_000
    ):
        super().__init__()
        # 1) input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        # 2) positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        # 3) transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False  # expecting [seq, batch, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # 4) output MLP
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        # project to d_model
        x = self.input_proj(x)                      # [B, S, d_model]
        # reshape for transformer: [S, B, d_model]
        x = x.permute(1, 0, 2)
        # add positional encoding
        x = self.pos_encoder(x)
        # transformer encoding
        x = self.transformer_encoder(x)             # [S, B, d_model]
        # back to [B, S, d_model]
        x = x.permute(1, 0, 2)
        # global average pooling over time
        x = x.mean(dim=1)                            # [B, d_model]
        # final feature projection
        out = self.output_mlp(x)                     # [B, output_dim]
        return out
    
class CNUNormalDataset(Dataset):
    """
    cnu_normal_0.npy ... cnu_normal_99.npy 를 읽어오는 Dataset
    각 .npy 파일은 (seq_len, input_dim=5) 형태라고 가정합니다.
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        # 폴더 내 .npy 파일 경로를 번호 순서대로 정렬
        files = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        files.sort(key=lambda fn: int(fn.split('_')[-1].split('.npy')[0]))
        self.paths = [os.path.join(root_dir, f) for f in files]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # NumPy array 로드 → Tensor 변환
        arr = np.load(self.paths[idx])      # shape: (seq_len, 5)
        return torch.from_numpy(arr).float()


class SlidingWindowDataset(Dataset):
    """
    CNUNormalDataset 위에 sliding window 를 씌우는 래퍼.
    - window_size: 윈도우 길이 W
    - stride: 다음 윈도우 시작 간격 S
    """
    def __init__(self, base_dataset: CNUNormalDataset, window_size: int, stride: int):
        self.base = base_dataset
        self.W = window_size
        self.S = stride
        # 각 윈도우가 (file_idx, start_idx) 로 매핑되도록 인덱스 목록 생성
        self.index_map = []
        for file_idx in range(len(self.base)):
            seq_len = np.load(self.base.paths[file_idx]).shape[0]
            # start 위치를 0, S, 2S, … 로 반복
            for start in range(0, seq_len, self.S):
                self.index_map.append((file_idx, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, start = self.index_map[idx]
        seq = np.load(self.base.paths[file_idx])    # (L, 5)
        end = start + self.W
        if end <= seq.shape[0]:
            window = seq[start:end]
        else:
            # 끝이 모자랄 때는 0으로 패딩
            pad_len = end - seq.shape[0]
            window = np.concatenate([
                seq[start:],
                np.zeros((pad_len, seq.shape[1]), dtype=seq.dtype)
            ], axis=0)
        return torch.from_numpy(window).float()      # (W, 5)
    

def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """
    가변 길이 시퀀스를 패딩하여 배치화
    batch: [(seq_len_i, 5), ...]
    반환: Tensor of shape [batch_size, max_seq_len, 5]
    """
    return pad_sequence(batch, batch_first=True, padding_value=0.0)

# ─── Dataset & DataLoader 생성 ──────────────────────────────────────
root = "/data/ess/data/HCT_CNU_100cycle/preprocessed_CNU/CNU_normal"
dataset = CNUNormalDataset(root_dir=root)

base_ds    = CNUNormalDataset(root)
win_ds     = SlidingWindowDataset(base_ds, window_size=4096, stride=2048)

dataloader = DataLoader(
    win_ds,
    batch_size=1,         
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)    


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TimeSeriesTransformer(
        input_dim=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        output_dim=10,
        dropout=0.1,
        max_len=100_000
    ).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    
    for epoch in tqdm(range(10)):
        for batch in tqdm(dataloader):
            
            # data = np.squeeze(batch,axis=0)
            # pd = npy_to_parquet(data)
            # breakpoint()
            # soc = calculate_soc_slope(pd)
            
            # print(soc)
            # batch: [B, S_max, 5]
            batch = batch.to(device)
            features = model(batch)   # → [B, 10]
            print(features.shape)
            # (예시) loss 계산 및 업데이트
            # loss = loss_fn(features, targets)
            # optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"[Epoch {epoch+1}] 완료")