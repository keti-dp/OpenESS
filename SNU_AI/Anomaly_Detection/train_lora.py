"""
AnomalyBERT
################################################

Reference:
    Yungi Jeong et al. "AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme" in ICLR Workshop, "Machine Learning for Internet of Things(IoT): Datasets, Perception, and Understanding" 2023.

Reference:
    https://github.com/Jhryu30/AnomalyBERT
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf

from tqdm import tqdm

from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter

import utils.config as config
from models.anomaly_transformer_lora import get_anomaly_transformer
from utils.dataset import ESS_dataset

from estimate import estimate, ess_score
import loralib as lora


def main(args):
    options = OmegaConf.load(args.options)
    options.update(args.__dict__)

    prev_file = torch.load(args.checkpoint, map_location='cpu')
    prev_options = prev_file['options']
    assert options.dataset != prev_options.dataset, 'Only for transfer learning, not resuming.'

    options = OmegaConf.merge(prev_options, options)

    # Load data.
    train_dataset = ESS_dataset(options=options, is_train=True)
    test_dataset = ESS_dataset(options=options, seed=0, is_train=False)
    options.data.deg_num = test_dataset.time_len // 86400
    test_data, test_label = test_dataset.get_test_data(deg_num=options.data.deg_num)

    # Define model.
    device = torch.device('cuda:{}'.format(args.gpu_id))
    model = get_anomaly_transformer(options=options.network,
                                    input_d_data=train_dataset.column_len,
                                    output_d_data=1,
                                    hidden_dim_rate=4.,
                                    positional_encoding=None,
                                    relative_position_embedding=True,
                                    transformer_n_head=8,
                                    rank=args.rank).to(device)
    
    # Load a checkpoint.
    loaded_weight = prev_file['model_state_dict']
    try:
        model.load_state_dict(loaded_weight, strict=False)
    except:
        loaded_weight['linear_embedding.weight'] = model.linear_embedding.weight
        loaded_weight['linear_embedding.bias'] = model.linear_embedding.bias
        model.load_state_dict(loaded_weight)
    lora.mark_only_lora_as_trainable(model)

    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    log_dir = os.path.join(config.LOG_DIR, time.strftime('%y%m%d%H%M%S_'+'lora_'+options.dataset, time.localtime(time.time())))
    options.log_dir = log_dir
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'state'))
    
    # hyperparameters save
    with open(os.path.join(log_dir, 'hyperparameters.yaml'), 'w') as f:
        OmegaConf.save(config=options, f=f)
    
    summary_writer = SummaryWriter(log_dir)
    torch.save(model, os.path.join(log_dir, 'model.pt'))

    # Train loss
    train_loss = nn.BCELoss().to(device)
    sigmoid = nn.Sigmoid().to(device)

    # Optimizer and scheduler
    max_iters = options.max_steps + 1
    lr = options.lr
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=max_iters,
                                  lr_min=lr*0.01,
                                  warmup_lr_init=lr*0.001,
                                  warmup_t=max_iters // 10,
                                  cycle_limit=1,
                                  t_in_epochs=False)
    
    # Start training.
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=options.batch_size)

    for i, batch in enumerate(tqdm(train_dataloader)):
        x, x_anomaly, _= batch

        # Process data.
        x = x.to(device)
        x_anomaly = x_anomaly.to(device)
        y = model(x).squeeze(-1)

        # Compute losses.
        loss = train_loss(sigmoid(y), x_anomaly)

        # Print training summary.
        if i % options.summary_steps == 0:
            with torch.no_grad():
                n_batch = options.batch_size
                pred = (sigmoid(y) > 0.5).int()
                x_anomaly = x_anomaly.bool().int()
                total_data_num = n_batch * train_dataset.data_seq_len
                
                acc = (pred == x_anomaly).int().sum() / total_data_num
                summary_writer.add_scalar('Train/Loss', loss.item(), i)
                summary_writer.add_scalar('Train/Accuracy', acc, i)
                
                model.eval()

                estimation = estimate(test_data, model, sigmoid, 1, n_batch, options.window_sliding, None, device)
                estimation = estimation[:, 0].cpu().numpy()
                model.train()
                
                best_eval = (0, 0, 0)
                best_rate = 0
                for rate in np.arange(0.001, 0.301, 0.001):
                    evaluation = ess_score(test_label, estimation, rate, False, False)
                    if evaluation[2] > best_eval[2]:
                        best_eval = evaluation
                        best_rate = rate
                summary_writer.add_scalar('Valid/Best Anomaly Rate', best_rate, i)
                summary_writer.add_scalar('Valid/Precision', best_eval[0], i)
                summary_writer.add_scalar('Valid/Recall', best_eval[1], i)
                summary_writer.add_scalar('Valid/ess', best_eval[2], i)
                
                print(f'iteration: {i} | loss: {loss.item():.10f} | train accuracy: {acc:.10f}')
                print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n')

            torch.save({'model_state_dict' : lora.lora_state_dict(model), 'options' : options}, os.path.join(log_dir, 'state/lora_state_dict_step_{}.pt'.format(i)))

        # Update gradients.
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip_norm)

        optimizer.step()
        scheduler.step_update(i)

    torch.save({'model_state_dict' : lora.lora_state_dict(model), 'options' : options}, os.path.join(log_dir, 'lora_state_dict.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", default='./configs/train_lora_config.yaml')
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str, help='load checkpoint file')
    parser.add_argument("--rank", default=1, type=int, help="rank to apply LoRA")
    args = parser.parse_args()

    main(args)
    # panli : /data/ess/output/Anomaly_Detection/logs/250620131607_ESS_panli/state_dict.pt
    # sionyu : /data/ess/output/Anomaly_Detection/logs/250818154904_ESS_sionyu/state_dict.pt - seed 0
    # gold : /data/ess/output/Anomaly_Detection/logs/250819131849_ESS_gold/state_dict.pt

    # sionyu to panli : /data/ess/output/Anomaly_Detection/logs/250819052124_lora_ESS_panli - seed 1000
    # sionyu to gold : /data/ess/output/Anomaly_Detection/logs/250819153537_lora_ESS_gold - seed 0
    # panli to sionyu : /data/ess/output/Anomaly_Detection/logs/250819193706_lora_ESS_sionyu X
    # panli to gold : /data/ess/output/Anomaly_Detection/logs/250819133243_lora_ESS_gold - seed 0
    # gold to sionyu : /data/ess/output/Anomaly_Detection/logs/250819194512_lora_ESS_sionyu X 
    # gold to panli : /data/ess/output/Anomaly_Detection/logs/250819194506_lora_ESS_panli - seed 1000