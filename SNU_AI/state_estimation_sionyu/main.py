import argparse
import yaml
import os

import utils
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from utils.util import str2bool, seed_everything, config_update
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


def ready(args):
    # config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # override config with parser
    for key, value in vars(args).items():
        config[key] = value

    seed_everything(config['seed'])


    # instruction
    print('Input Columns : {}'.format(config['input_cols']))
    print('Target Column : {}'.format(config['target_col']))
    print('Model : {}'.format(config['model']))

    # makedir
    ymdt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # year month date time
    ymdt = ymdt.replace(' ', '_')
    folder_name = config['output_dir'] + config['output_name'] + ymdt

    # 시간이 겹쳤을때. 우리는 한꺼번에 많이 실험돌릴꺼라 파일 이름이 겹칠수 있음. 그래서 인덱스를 설정해주겠음.
    if not os.path.exists(folder_name):
        os.makedirs(folder_name +'/ckpt', exist_ok=False)
    else:
        for index in range(10000):
            if os.path.exists( folder_name + '_' + str(index) ):
                continue
            else:
                folder_name = folder_name + '_' + str(index)
                os.makedirs(folder_name +'/ckpt', exist_ok=False)
                break

    # log
    log_file = folder_name + '/log.txt'
    config['folder_name'] = folder_name
    config['log_file'] = log_file

    # model/dataset/dataloader
    model = utils.CustomModel(config=config)
    train_loader, val_loader, test_loader = utils.build_loader(config)

    # yaml로 다시 저장
    with open(folder_name + '/config.yaml','w') as f:
        yaml.dump(config, f)

    return config, model, train_loader, val_loader, test_loader


def train(config, model, train_loader, val_loader, test_loader):

    # optim/loss_fn
    decay = config['decay']
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: decay ** epoch)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=20,
                                          cycle_mult=1.0,
                                          max_lr=1e-4,
                                          min_lr=0,
                                          warmup_steps=15,
                                          gamma=decay)
    loss_fn = getattr(nn, config['loss_fn'])()

    device = torch.device('cuda')
    train_loss_log = []
    val_loss_log = []
    test_loss_log = []
    best_acc = 100
    for epoch in range(config['epochs']):

        # train
        train_loss = 0
        for i, (input, target) in enumerate(train_loader):
            # data = B x C x time
            model.train()
            optimizer.zero_grad()
            input, target = input.reshape(-1, input.size()[-2], input.size()[-1]), target.reshape(-1, target.size()[-1])
            input, target = input.to(device), target.to(device)
            model = model.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            train_loss += loss
            loss.backward()
            optimizer.step()

        train_loss_log.append( train_loss/len(train_loader) )
        scheduler.step()

        # validation
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for i, (input, target) in enumerate(val_loader):
                input, target = input.to(device), target.to(device)
                output = model(input)
                val_loss += loss_fn(output, target)
            val_loss_log.append( val_loss/len(val_loader) )

            print(f"Epoch {epoch+1} train {config['loss_fn']} : {loss:.4f} | val {config['loss_fn']} : {val_loss:.4f}")

            # log
            utils.write_log(config['log_file'], f"Epoch {epoch+1} train {config['loss_fn']} : {loss:.4f} | val {config['loss_fn']} : {val_loss:.4f} \n")

            if val_loss <= best_acc:
                best_acc = val_loss
                torch.save(model.state_dict(), config['folder_name']+'/ckpt'+'/checkpoint.pt')

                # torch.save(model.state_dict(), config['output_dir'] + config['output_name']+'/ckpt'+'/{}.pt'.format(config['model']))
                # torch.save(model.state_dict(), config['output_dir'] + config['output_name']+'/ckpt'+'/{}_epoch_{}.pt'.format(config['model'], epoch+1))

    # # test
    # with torch.no_grad():
    #     model.eval()
    #     test_loss = 0
    #     for i, (input, target) in enumerate(test_loader):
            # input, target = input.to(device), target.to(device)
    #         output = model(input)
    #         test_loss += loss_fn(output, target)
    #     test_loss_log.append( test_loss/len(test_loader) )
    #     print('test {} : {}'.format(config['loss_fn'], test_loss))
    #     utils.write_log(config['log_file'], 'test {} : {}'.format(config['loss_fn'], test_loss))
    
    
    
    # Loss plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    train_loss_plot = [tensor.cpu().detach().numpy() for tensor in train_loss_log]
    val_loss_plot = [tensor.cpu().detach().numpy() for tensor in val_loss_log]
    test_loss_plot = [tensor.cpu().detach().numpy() for tensor in test_loss_log] * config['epochs']
    
    ax.plot(train_loss_plot, linewidth=2, c='royalblue', label='Train loss')
    ax.plot(val_loss_plot, linewidth=2, c='crimson', label='Val loss')
    ax.plot(test_loss_plot, linewidth=2, c='darkviolet', label='Test loss')
    ax.set_xticks(np.linspace(0, config['epochs'], 11))
    ax.set_title('Model : {}'.format(config['model']), fontdict={'family': 'Sans-serif', 'weight': 'bold', 'size': 14})
    ax.grid()
    ax.legend()
    fig.savefig(os.path.join(config['folder_name'], 'Loss_plot.png'))
    plt.close()


    # save
    torch.save(torch.tensor(val_loss_log), os.path.join(config['folder_name'],'val_loss_log.pt'))
    torch.save(torch.tensor(train_loss_log), os.path.join(config['folder_name'],'train_loss_log.pt'))


if __name__ == '__main__':

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=str, default='sionyu')
    parser.add_argument('--loss_fn', type=str, default='L1Loss')
    parser.add_argument('--data_per_oneday', type=int, default=1)


    parser.add_argument('--model', type=str, default='res10')
    parser.add_argument('--seconds', type=int, default=25)

    parser.add_argument('--epochs', type=int, default='100')

    parser.add_argument('--decay', type=float, default=0.98)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--debug', type=str2bool, default=False)
    # parser.add_argument('--', type=str, default='')

    args = parser.parse_args()


    config, model, train_loader, val_loader, test_loader = ready(args)
    train(config, model, train_loader, val_loader, test_loader)

    




    

    
