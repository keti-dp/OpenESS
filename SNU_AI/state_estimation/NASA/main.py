import os
import argparse
import logging
import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from dataloader import get_dataloader, denormalize_vi
from models import EncDec
from models.encoders import CNN_encoder

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# This is for the demo at 0717

def main(args: argparse.Namespace):
    if args.seed is not None:
        seed_everything(int(args.seed))
    # Dataloader
    train_battery = [int(item) for item in args.train_battery.split(',')]
    test_battery = [int(item) for item in args.test_battery.split(',')]
    train_loader, test_loader = get_dataloader(
        args.batch_size, args.input_process_type, args.output_type, train_battery, test_battery
    )

    # Get model
    model = EncDec(args.encoder, args.decoder, args.input_len, output_length=9001)
    logging.info(f'Model Params: {sum(p.numel() for p in model.parameters()) / 1024 / 1024:.3f}M')
    # Load model if needed
    if args.load_exp :
        enc_ckpt = f'./logs/{args.load_exp}/ckpt/encoder.pt'
        dec_ckpt = f'./logs/{args.load_exp}/ckpt/decoder.pt'
        model.encoder.load_state_dict(torch.load(enc_ckpt))
        model.decoder.load_state_dict(torch.load(dec_ckpt))

    # Prepare to train
    device = torch.device(f'cuda:{args.gpu}')
    model = model.to(device)
    loss_fn_1 = nn.MSELoss(reduction='sum')
    #loss_fn_2 = nn.MSELoss(reduction='sum')
    optimizer_enc = torch.optim.Adam(model.encoder.parameters(), lr=1e-3)
    optimizer_dec = torch.optim.Adam(model.decoder.parameters(), lr=1e-3)
    lr_lambda = lambda epoch: max(0.995 ** epoch, 0.1)
    scheduler_enc = torch.optim.lr_scheduler.LambdaLR(optimizer_enc, lr_lambda)
    # scheduler_enc = create_lr_scheduler_with_warmup(scheduler_enc,
    #                                         warmup_start_value=0,
    #                                         warmup_end_value=1e-5,
    #                                         warmup_duration=20)
    
    # scheduler_enc(None)
    # scheduler_enc = CosineAnnealingWarmupRestarts(optimizer=optimizer_enc,
    #                                               first_cycle_steps=10,
    #                                               cycle_mult=1.0,
    #                                               max_lr=1e-4,
    #                                               min_lr=0,
    #                                               warmup_steps=7,
    #                                               gamma=0.9)
    scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec, lr_lambda)

    # Start! 
    start = datetime.now()
    soh_mae_list = []
    ri_rmse_list = []
    for epoch in range(args.epoch):

        # Train
        cum_loss1 = 0.
        #cum_loss2 = 0.
        model.train()
        # i = 0
        for soh, drive, ri_curve in train_loader:
            # i += 1
            soh = soh.to(device)
            drive = drive.to(device)
            ri_curve = ri_curve.to(device)
            soh_pred, ri_pred = model(drive, soh)
            loss1 = loss_fn_1(soh, soh_pred)
            #loss2 = loss_fn_2(ri_curve, ri_pred)
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            loss1.backward()
            #loss2.backward()
            optimizer_enc.step()
            optimizer_dec.step()
            cum_loss1 += loss1.item()
            #cum_loss2 += loss2.item()
            ################################################################
            
            # if (epoch + 1) % 20 == 0:
            #     train_true_list = []
            #     train_pred_list = []
            #     test_true_list = []
            #     test_pred_list = []

            #     with torch.no_grad():
                    
            #         for soh, drive, _ in train_loader:
            #             soh_pred = model.encoder(drive.to(device))
            #             train_true_list.append(soh.numpy())
            #             train_pred_list.append(soh_pred.detach().cpu().numpy())
                        
            #         for soh, drive, _ in test_loader:
            #             soh_pred = model.encoder(drive.to(device))
            #             test_true_list.append(soh.numpy())
            #             test_pred_list.append(soh_pred.detach().cpu().numpy())
                
            #     _train_true = np.concatenate(train_true_list, axis=0)
            #     _train_pred = np.concatenate(train_pred_list, axis=0).reshape(-1)
            #     _test_true = np.concatenate(test_true_list, axis=0)
            #     _test_pred = np.concatenate(test_pred_list, axis=0).reshape(-1)
            #     train_r_sq = r2_score(_train_true, _train_pred)
            #     test_r_sq = r2_score(_test_true, _test_pred)
            
            
            #     fig = plt.figure(figsize=(12, 6))
            #     plt.title(f'SOH estimation result and R^2 score, Test R^2 : {test_r_sq:.4f}', fontdict={'weight': 'bold', 'size': 14})
            #     plt.axis('off')
            #     ax = fig.add_subplot(121)
            #     ax.scatter(_train_true, _train_pred, s=4, label=f'Train R^2: {train_r_sq:.4f}', c='indianred')
            #     ax.plot([0.4, 1], [0.4, 1], c='black', alpha=0.7, linewidth=3)
            #     ax.set_xlim([0.35, 1.05])
            #     ax.set_xlabel('True SOH', fontdict={'weight': 'bold', 'size': 14})
            #     ax.set_ylim([0.35, 1.05])
            #     ax.set_ylabel('Pred SOH', fontdict={'weight': 'bold', 'size': 14})
            #     ax.grid()
            #     ax.legend(loc='upper left', prop={'size': 12})

            #     ax = fig.add_subplot(122)
            #     ax.scatter(_test_true, _test_pred, s=4, label=f'Test R^2: {test_r_sq:.4f}', c='royalblue')
            #     ax.plot([0.4, 1], [0.4, 1], c='black', alpha=0.7, linewidth=3)
            #     ax.set_xlim([0.35, 1.05])
            #     ax.set_xlabel('True SOH', fontdict={'weight': 'bold', 'size': 14})
            #     ax.set_ylim([0.35, 1.05])
            #     ax.set_ylabel('Pred SOH', fontdict={'weight': 'bold', 'size': 14})
            #     ax.grid()
            #     ax.legend(loc='upper left', prop={'size': 12})

            #     fig.savefig('figures/SOH_result_R^2.png', dpi=300)
            #     plt.show()
                
            #     fig.savefig(os.path.join(args.img_dir, f'Epoch_{epoch}_Iter_{i}.png'))
            #     plt.close()
            
            ####################################################################
            
        cum_loss1 /= len(train_loader.dataset)
        #cum_loss2 /= len(train_loader.dataset)
        scheduler_enc.step()
        scheduler_dec.step()

        # Test
        cum_soh_mae = 0.
        cum_ri_rmse = 0.
        model.eval()
        with torch.no_grad():
            for soh, drive, ri_curve in test_loader:
                soh = soh.to(device)
                drive = drive.to(device)
                soh_pred, ri_pred = model(drive, soh)
                soh_mae = np.abs(soh.detach().cpu().numpy() - soh_pred.detach().cpu().numpy()).sum()
                ri_rmse = ((ri_curve.numpy() - ri_pred.detach().cpu().numpy()) ** 2).sum()
                cum_soh_mae += soh_mae
                cum_ri_rmse += ri_rmse
        cum_soh_mae /= len(test_loader.dataset)
        cum_ri_rmse /= len(test_loader.dataset)
        cum_ri_rmse = np.sqrt(cum_ri_rmse)
        soh_mae_list.append(cum_soh_mae)
        ri_rmse_list.append(cum_ri_rmse)

        # Logging
        logging.info(
            f'Epoch {epoch+1} lr {scheduler_enc.get_last_lr()[0]:.3e} [Test] [SOH (MAE)] {cum_soh_mae:.3e} [Ri (RMSE)] {cum_ri_rmse:.3e} '\
            f'Elapsed {datetime.now() - start}'
        )
        
        
        '''
        # Visualize
        if (epoch + 1) % 20 == 0:
            _soh, _drive, _ri_curve = test_loader.dataset.__getitem__(np.random.randint(low=0, high=len(test_loader.dataset)))
            with torch.no_grad():
                soh_pred, ri_pred = model(
                    torch.from_numpy(_drive).unsqueeze(0).to(device),
                    torch.tensor([_soh], dtype=torch.float32).unsqueeze(0).to(device),
                )
            fig = plt.figure(figsize=(12, 8))
            true_soh = _soh * 100
            pred_soh = soh_pred.item() * 100
            plt.title(
                f'Real SOH: {true_soh:.2f}%, Pred SOH: {pred_soh:.2f}%, Gap: {pred_soh - true_soh:.2f}%',
                fontdict={'family': 'Sans-serif', 'weight': 'bold', 'size': 14},
            )
            plt.axis('off')
            drive_t = np.arange(len(_drive))
            drive_vi = denormalize_vi(np.expand_dims(_drive, axis=0)).squeeze(axis=0)
            # (1) Drive V
            ax = fig.add_subplot(221)
            ax.scatter(drive_t, drive_vi[:, 0], label='Drive_V', s=4, c='blue')
            ax.set_ylim([drive_vi[:, 0].min() - 0.1, drive_vi[:, 0].max() + 0.1])
            ax.grid()
            ax.legend()
            # (2) Drive I
            ax = fig.add_subplot(222)
            ax.scatter(drive_t, drive_vi[:, 1], label='Drive_I', s=4, c='red')
            ax.set_ylim([drive_vi[:, 1].min() - 0.1, drive_vi[:, 1].max() + 0.1])
            ax.grid()
            ax.legend()
            # (3) R_i true vs pred
            ax = fig.add_subplot(212)
            ri_mse = ((_ri_curve - ri_pred.detach().cpu().numpy()) ** 2).sum()
            ri_pred = ri_pred.detach().cpu().numpy()[0, :]
            ax.set_title(f'RMSE of R_i estimation: {ri_mse:.3f}', fontdict={'family': 'Sans-serif', 'weight': 'bold', 'size': 14})
            soc = np.linspace(0.05, 0.95, 9001)
            ax.scatter(soc, _ri_curve, label='real R_i', s=4, c='green', alpha=0.7)
            ax.scatter(soc, ri_pred, label='pred R_i', s=4, c='orange', alpha=0.7)
            ax.grid()
            ax.legend(prop={'size': 13})
            fig.savefig(os.path.join(args.img_dir, f'Epoch_{epoch+1}.png'))
            plt.close()
        '''
    
    # SOH MAE plot for transfer learning
    soh_mae_plot = open(os.path.join(args.soh_mae_dir, 'soh_mae_plot.txt'), 'w')
    for soh in soh_mae_list:
        soh_mae_plot.write(str(soh))
        soh_mae_plot.write('\n')
    soh_mae_plot.close()    
    
        
    # Save model
    torch.save(model.encoder.state_dict(), os.path.join(args.ckpt_dir, 'encoder.pt'))
    torch.save(model.decoder.state_dict(), os.path.join(args.ckpt_dir, 'decoder.pt'))
    logging.info('Model saved!')

    # MAE, RMSE plot
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.set_ylim([0, 0.5])
    ax.axhline(0.03, 0, 1, color='lightgray', linestyle='--', linewidth=2)
    ax.plot(soh_mae_list, linewidth=2, c='blueviolet', label='Test MAE (SOH)')
    ax.set_xticks(np.linspace(0, args.epoch, 11))
    ax.grid()
    ax.legend()
    fig.savefig(os.path.join(args.img_dir, f'MAE.png'))
    plt.close()
    
    # fig = plt.figure(figsize=(10, 4))
    # ax = fig.add_subplot(111)
    # ax.plot(ri_rmse_list, linewidth=2, c='firebrick', label='Test RMSE (R_i)')
    # ax.set_xticks(np.linspace(0, args.epoch, 11))
    # ax.grid()
    # ax.legend()
    # fig.savefig(os.path.join(args.img_dir, f'RMSE.png'))
    plt.close()

    return

# python main.py --exp=1718 --train_battery=17 --test_battery=18 --epoch=400 --seed=0

# python main.py --exp=1719 --train_battery=17 --test_battery=19 --epoch=400 --seed=0

# python main.py --exp=1819 --train_battery=18 --test_battery=19 --epoch=400 --seed=0

# python main.py --exp=1720 --train_battery=17 --test_battery=20 --epoch=400 --seed=0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--load_exp', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--encoder', type=str, choices=['mlp', 'cnn', 'rnn', 'trn'], default='cnn')
    parser.add_argument('--decoder', type=str, choices=['vv', 'vf', 'fv', 'ff'], default='vv')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--input-len', type=int, default=64)
    parser.add_argument('--seed', default=None)
    

    # 사용예시
    # python main.py --train_battery=17,18,19
    # python main.py --train_battery='17,18,19'
    parser.add_argument('--train_battery', type=str, default='17,18,19')
    parser.add_argument('--test_battery', type=str, default='20')


    args = parser.parse_args()

    # Input / Output processing type
    if args.encoder != 'trn':   
        args.input_process_type = 'cut' + str(args.input_len)
    else:
        args.input_process_type = 'mask' + str(args.input_len)
    args.output_type = 'vec'

    # Set GPU num
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'

    # Logging
    args.log_dir = os.path.join('logs', args.exp) 
    os.makedirs(args.log_dir, exist_ok=True)
    args.tb_dir = os.path.join(args.log_dir, 'tb')
    os.makedirs(args.tb_dir, exist_ok=True)
    args.ckpt_dir = os.path.join(args.log_dir, 'ckpt')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.img_dir = os.path.join(args.log_dir, 'img')
    os.makedirs(args.img_dir, exist_ok=True)
    args.soh_mae_dir = os.path.join(args.log_dir, 'soh_mae')
    os.makedirs(args.soh_mae_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'{os.path.join(args.log_dir, "train_log.txt")}', mode='w'),
        ],
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )

    
    
    main(args)
