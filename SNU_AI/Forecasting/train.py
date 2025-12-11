import os
import glob
import math
import random
import json
import argparse
import numpy as np

from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

from dataset import build_loader

from model import lstm_generator

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

# parser.add_argument('--custom', default=None, type=int)
# parser.add_argument('--custom', default=None, type=str2bool)




def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    # loss 로그 파일 경로
    loss_txt = os.path.join(args.save_path, "loss.txt")

    # 사용한 설정 저장(재현성용)
    with open(os.path.join(args.save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 1.loader
    train_dir = args.data_dir
    test_dir = args.data_dir
    train_loader, test_loader = build_loader(train_dir, test_dir, args.batch_size)

    # 2.model - 종현
    device = torch.device('cuda:{}'.format(args.gpu_id))
    model = lstm_generator(input_dim=3,   # [SoC, sin, cos]
                           hidden=256, 
                           layers=args.num_layers, 
                           dropout=args.dropout, 
                           output_dim=1).to(device)

    # 3.optimizer - 종현
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == "adamw": 
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss == 'mae':
        loss_fn = nn.L1Loss()
    scaler = GradScaler()

    # 4.Generation train code - 종현
    # train
    best_val = float('inf')
    for epoch in range(args.epoch):
        print(f"\nEpoch {epoch+1}/{args.epoch}")
        total_train_loss = 0.0
        n_train_samples = 0
        model.train()

        lr = adjust_learning_rate(optimizer, epoch, args)

        # train loop (CHUNKED)
        chunk_len = getattr(args, "chunk_len", 2048)

        train_iter = tqdm(train_loader, desc=f"Train {epoch+1}", leave=False)
        total_train_loss = 0.0
        total_train_elems = 0  # = 배치*시간 길이 합
        last_sub_loss = 0.0

        for step, (today, tomorrow) in enumerate(train_iter):
            today = today.to(device, non_blocking=True).float()     # [B,T,1]
            tomorrow = tomorrow.to(device, non_blocking=True).float()
            assert today.dim() == 3 and today.size(-1) == 1, f"expected [B,T,1], got {today.shape}"

            # 배치 준비 직후 (today: [B,T,1])
            # 모델에게 시간정보 추가.
            T = today.size(1)
            t = torch.arange(T, device=today.device).float().view(1, T, 1)
            sin_t = torch.sin(2*math.pi*t/T)
            cos_t = torch.cos(2*math.pi*t/T)
            today = torch.cat([today, sin_t.expand_as(today), cos_t.expand_as(today)], dim=-1)  # [B,T,3]. today with time


            bs = today.size(0)
            h = None
            optimizer.zero_grad(set_to_none=True)

            # --- TBPTT: 청크마다 forward/backward/step, hidden 전달+detach ---
            for s, today_chunk in iter_chunks(today, chunk_len):
                target_chunk = tomorrow[:, s:s+chunk_len, :]        # [B,t,1] (t<=chunk_len)
                delta_target = (target_chunk - today_chunk[:,:,:1]).float()     # 내일-오늘

                with autocast(dtype=torch.float16):
                    delta_pred, h = model(today_chunk, h)               # 모델은 Δ만 출력 (out_dim=1 그대로)
                with torch.cuda.amp.autocast(enabled=False):            # FP32 손실
                    # (선택1) Δ만의 손실
                    loss_delta = loss_fn(delta_pred.float(), delta_target) # TODO # * args.output_scale?
                    # (선택2) 최종 예측(pred = today + Δ) 도 같이 맞춤
                    pred_chunk = today_chunk[:,:,:1].float() + delta_pred.float()
                    loss_abs = loss_fn(pred_chunk, target_chunk.float())

                    # 조합(가중치는 상황에 맞게; 1.0/0.5 정도로 시작)
                    sub_loss = args.lambda1 * loss_abs + args.lambda2 * loss_delta


                with torch.no_grad():
                    print(
                        "MSE(today, target_chunk):", ((today_chunk[:, :, :1] - target_chunk)**2).mean().item(),
                        "MSE(pred , target_chunk):", ((pred_chunk         - target_chunk)**2).mean().item(),
                        "MSE(today , pred):", ((pred_chunk         - today_chunk[:, :, :1])**2).mean().item(),
                    )
                    print(
                        "Δ_target abs mean:", (delta_target.abs().mean().item()),
                        "Δ_pred   abs mean:", (delta_pred.abs().mean().item()),
                    )

                

                # 역전파 & 청크마다 1회 step (그래프 누수 방지)
                scaler.scale(sub_loss).backward()
                scaler.unscale_(optimizer)                          # (선택) grad clipping 전처리
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # (선택) 클리핑

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # 다음 청크로 넘어가기 전에 hidden을 그래프에서 분리
                h = (h[0].detach(), h[1].detach()) if isinstance(h, tuple) else h.detach()

                # 가중 평균용 집계 (배치*시간길이)
                elems = bs * today_chunk.size(1)
                total_train_loss += sub_loss.item() * elems
                total_train_elems += elems
                last_sub_loss = sub_loss.item()

            train_iter.set_postfix(loss=f"{last_sub_loss:.2e}", lr=f"{lr:.2e}")

        train_loss = total_train_loss / max(1, total_train_elems)

        # validation loop (TBPTT, CHUNKED)
        total_val_loss = 0.0
        total_val_elems = 0
        model.eval()
        val_iter = tqdm(test_loader, desc=f"Val {epoch+1}", leave=False)
        last_val_sub_loss = 0.0

        with torch.inference_mode():
            for today, tomorrow in val_iter:
                today = today.to(device, non_blocking=True).float()
                tomorrow = tomorrow.to(device, non_blocking=True).float()
                bs = today.size(0)

                # 배치 준비 직후 (today: [B,T,1])
                # 모델에게 시간정보 추가.
                T = today.size(1)
                t = torch.arange(T, device=today.device).float().view(1, T, 1)
                sin_t = torch.sin(2*math.pi*t/T)
                cos_t = torch.cos(2*math.pi*t/T)
                today = torch.cat([today, sin_t.expand_as(today), cos_t.expand_as(today)], dim=-1)  # [B,T,3]. today with time

                h = None  # 배치마다 hidden 초기화
                for s, today_chunk in iter_chunks(today, chunk_len):
                    target_chunk = tomorrow[:, s:s+chunk_len, :]

                    with autocast(dtype=torch.float16):
                        delta_pred, h = model(today_chunk, h)

                    # 최종 예측은 today + Δ
                    pred_chunk = today_chunk[:, :, :1].float() + delta_pred.float()
                    sub_loss   = loss_fn(pred_chunk, target_chunk.float())

                    # (inference_mode라 그래프는 안 쌓이지만 형태 통일을 위해 detach)
                    h = (h[0].detach(), h[1].detach()) if isinstance(h, tuple) else h.detach()

                    elems = bs * today_chunk.size(1)
                    total_val_loss += sub_loss.item() * elems
                    total_val_elems += elems
                    last_val_sub_loss = sub_loss.item()

                val_iter.set_postfix(loss=f"{last_val_sub_loss:.2e}", lr=f"{lr:.2e}")

        val_loss = total_val_loss / max(1, total_val_elems)

        print(f"  train_loss: {train_loss:.2e} | val_loss: {val_loss:.2e} | lr: {lr:.2e}")

        # loss.txt 기록
        if epoch == 0 and not os.path.exists(loss_txt):
            with open(loss_txt, "w") as f:
                f.write("epoch\ttrain_loss\tval_loss\tlr\n")
        with open(loss_txt, "a") as f:
            f.write(f"{epoch+1}\t{train_loss:.2e}\t{val_loss:.2e}\t{lr:.6e}\n")

        ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "args": vars(args),
            }
        
        # latest 체크포인트 저장
        latest_path = os.path.join(args.save_path, "latest.pth.tar")
        torch.save(ckpt, latest_path)
        print(f"Latest Model saved → {latest_path}")

        # best 체크포인트 저장
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_path = os.path.join(args.save_path, "best.pth.tar")
            torch.save(ckpt, best_path)
            print(f"Best Model saved → {best_path}")


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        lr = args.lr * float(epoch + 1) / max(1, args.warmup_epochs)
    else:
        t = (epoch - args.warmup_epochs) / max(1, args.epoch - args.warmup_epochs)
        t = max(0.0, min(1.0, t))
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr

def iter_chunks(x, chunk_len):
    """x: [B, T, C] -> (start, x[:, start:start+chunk_len, :]) 생성"""
    T = x.size(1)
    for s in range(0, T, chunk_len):
        yield s, x[:, s:s+chunk_len, :]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--data_dir", default=False, type=str)
    parser.add_argument('--seed', default=None, type=int)

    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--save_path", default="./logs", type=str)

    parser.add_argument("--optimizer", default='adamw', type=str, choices=['adamw', 'sgd'])
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--loss", default='mse', type=str, choices=['mse', 'mae'])

    parser.add_argument("--warmup_epochs", default=5, type=int)
    parser.add_argument("--chunk_len", default=1024, type=int)
    parser.add_argument("--output_scale", default=100., type=float)

    parser.add_argument("--lambda1", default=0., type=float)
    parser.add_argument("--lambda2", default=1., type=float)

    parser.add_argument("--start", default=55000, type=int)
    parser.add_argument("--end", default=82000, type=int)

    args = parser.parse_args()

    main(args)
