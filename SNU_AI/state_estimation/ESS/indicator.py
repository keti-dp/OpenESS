import argparse
import yaml
import os
import random
from datetime import datetime
from tqdm import tqdm

import utils
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from IPython.display import Image
from IPython.display import display


from utils.util import str2bool, seed_everything, config_update, config_update_indicator, log2png, log2gif
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import main

def ready(args):
    # config
    with open('config_indicator.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # override config with parser
    config = config_update_indicator(config, args)

    common, base_config, config1, config2 = config['common'], config['base'], config['phase1'], config['phase2']


    # set seed
    seed_everything(common['seed'])

    # makedir
    ymdt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # year month date time
    ymdt = ymdt.replace(' ', '_')
    folder_name = common['output_dir'] + common['output_name'] + ymdt

    # 시간이 겹쳤을때. 우리는 한꺼번에 많이 실험돌릴꺼라 파일 이름이 겹칠수 있음. 그래서 인덱스를 설정해주겠음.
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=False)
    else:
        for index in range(10000):
            if os.path.exists( folder_name + '_' + str(index) ):
                continue
            else:
                folder_name = folder_name + '_' + str(index)
                # os.makedirs(folder_name +'/ckpt', exist_ok=False)
                break
        os.makedirs(folder_name, exist_ok=False)
        

    # log
    common['folder_name'] = folder_name
    base_config['log_file'] = folder_name + '/base_log.txt'
    config1['log_file'] = folder_name + '/phase1_log.txt'
    config2['log_file'] = folder_name + '/phase2_log.txt'

    # yaml로 다시 저장
    config['common'], base_config, config['phase1'], config['phase2'] = common,base_config, config1, config2
    with open(folder_name + '/config.yaml','w') as f:
        yaml.dump(config, f)

    base_model = utils.CustomModel(config=base_config)
    indi_model = utils.CustomModel(config=config1)
    eval_model = utils.CustomModel(config=config2)

    base_config.update(common)
    config1.update(common)
    config2.update(common)

    base_loaders = utils.build_loader(base_config)
    phase1_loaders = utils.build_loader(config1)
    phase2_loaders = utils.build_loader(config2)

    base_materials = [base_config, base_model, base_loaders]
    phase1_materials = [config1, indi_model, phase1_loaders]
    phase2_materials = [config2, eval_model, phase2_loaders]


    return config, base_materials, phase1_materials, phase2_materials
    # return config, indi_model, eval_model, phase1_loaders, phase2_loaders


def base_train(base_materials):
    print('Base Training Start', '-'*50)

    config, model, loaders = base_materials
    train_loader, val_loader, test_loader = loaders

    result_folder = config['folder_name'] + '/base'
    loss_log = result_folder + '/loss_log.txt'

    os.makedirs(result_folder+'/ckpt', exist_ok=False)
    # os.makedirs(result_folder+'/ckpt', exist_ok=False)


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
    output_log = []
    pbar = tqdm(range(config['epochs']),
                    total = len(range(config['epochs'])),
                    leave=True
                    )
    for epoch in pbar:

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
            one_epoch_output_log = []
            model.eval()
            val_loss = 0
            for i, (input, target) in enumerate(val_loader):
                input, target = input.reshape(-1, input.size()[-2], input.size()[-1]), target.reshape(-1, target.size()[-1])
                input, target = input.to(device), target.to(device)
                output = model(input)
                one_epoch_output_log.append(output)
                val_loss += loss_fn(output, target)

        output_log.append(torch.concat(one_epoch_output_log))
        val_loss /= len(val_loader)
        val_loss_log.append( val_loss )

        


        msg = f"Epoch {epoch+1} train {config['loss_fn']} : {loss:.4f} | val {config['loss_fn']} : {val_loss:.4f}"

        # log
        utils.write_log(loss_log,  msg + "\n")

        # print(msg)
        pbar.set_description(msg)


        if val_loss <= best_acc:
            best_acc = val_loss
            torch.save(model.state_dict(), result_folder+'/ckpt/base_{}.pt'.format(config['model']))


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
    phase = 'base'
    path = os.path.join(config['folder_name'], phase )
    log2png(config, path=path, name=f'{phase}_loss', phase='phase', title=f'{phase}_loss',
                        train_loss_log=train_loss_log, val_loss_log=val_loss_log)
    log2gif(config, path=path, name=f'{phase}_loss', phase='phase', title=f'{phase}_loss',
                        train_loss_log=train_loss_log, val_loss_log=val_loss_log)

    # save
    torch.save(torch.tensor(val_loss_log), os.path.join(result_folder,'base_val_loss_log.pt'))

    output_log = torch.concat(output_log)
    base_results = [config, model, loaders, output_log]
    return base_results


def phase1_train(phase1_materials):
    print('Phase1 Training Start', '-'*50)

    config, model, loaders = phase1_materials
    train_loader, val_loader, test_loader = loaders

    result_folder = config['folder_name'] + '/phase1'
    loss_log = result_folder + '/loss_log.txt'

    os.makedirs(result_folder+'/ckpt', exist_ok=False)



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
    output_log = []
    pbar = tqdm(range(config['epochs']),
                    total = len(range(config['epochs'])),
                    leave=True
                    )
    for epoch in pbar:

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
            one_epoch_output_log = []
            for i, (input, target) in enumerate(val_loader):
                input, target = input.reshape(-1, input.size()[-2], input.size()[-1]), target.reshape(-1, target.size()[-1])
                input, target = input.to(device), target.to(device)
                output = model(input)
                one_epoch_output_log.append(output)
                val_loss += loss_fn(output, target)

        output_log.append(torch.concat(one_epoch_output_log))
        val_loss /= len(val_loader)
        val_loss_log.append( val_loss )



        msg = f"Phase1 : Epoch {epoch+1} train {config['loss_fn']} : {loss:.4f} | val {config['loss_fn']} : {val_loss:.4f}"

        # log
        utils.write_log(loss_log,  msg + "\n")

        # print(msg)
        pbar.set_description(msg)

        if val_loss <= best_acc:
            best_acc = val_loss
            torch.save(model.state_dict(), result_folder+'/ckpt'+'/phase1_{}.pt'.format(config['model']))

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
    phase = 'phase1'
    path = os.path.join(config['folder_name'], phase )
    log2png(config, path=path, name=f'{phase}_loss', phase='phase', title=f'{phase}_loss',
                        train_loss_log=train_loss_log, val_loss_log=val_loss_log)
    log2gif(config, path=path, name=f'{phase}_loss', phase='phase', title=f'{phase}_loss',
                        train_loss_log=train_loss_log, val_loss_log=val_loss_log)

    # save
    torch.save(torch.tensor(val_loss_log), os.path.join(result_folder,'phase1_val_loss_log.pt'))


    model.eval()
    output_log = torch.concat(output_log)
    phase1_results = [config, model, loaders, output_log]
    return phase1_results


def phase2_train(phase1_results, phase2_materials):
    print('Phase 2 Training Start', '-'*50)

    _, indi_model, _, _ = phase1_results

    config, eval_model, loaders = phase2_materials
    train_loader, val_loader, test_loader = loaders

    result_folder = config['folder_name'] + '/phase2'
    loss_log = result_folder + '/loss_log.txt'

    os.makedirs(result_folder+'/ckpt', exist_ok=False)



    # optim/loss_fn
    decay = config['decay']
    optimizer = optim.SGD(eval_model.parameters(), lr=1e-5, momentum=0.9)
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
    output_log = []

    pbar = tqdm(range(config['epochs']),
                    total = len(range(config['epochs'])),
                    leave=True
                    )
    for epoch in pbar:

        # train
        train_loss = 0
        for i, (input, target) in enumerate(train_loader):
            
            # data = B x C x time
            optimizer.zero_grad()
            input, target = input.reshape(-1, input.size()[-2], input.size()[-1])[:,:-1], target.reshape(-1, target.size()[-1])   # indicator (ex. O-OCV) should be located at last channel
            input, target = input.to(device), target.to(device)
            indicator_col = indi_model(input).repeat(1, input.shape[-1]).view(input.shape[0],1,input.shape[-1])
            input = torch.concat([input, indicator_col], dim=1)


            eval_model.train()
            eval_model = eval_model.to(device)
            output = eval_model(input)
            loss = loss_fn(output, target)
            train_loss += loss
            loss.backward()
            optimizer.step()

        train_loss_log.append( train_loss/len(train_loader) )
        scheduler.step()

        # validation
        with torch.no_grad():
            eval_model.eval()
            val_loss = 0
            one_epoch_output_log = []
            for i, (input, target) in enumerate(val_loader):
                input, target = input.reshape(-1, input.size()[-2], input.size()[-1])[:,:-1], target.reshape(-1, target.size()[-1])   # indicator (ex. O-OCV) should be located at last channel
                input, target = input.to(device), target.to(device)
                indicator_col = indi_model(input).repeat(1, input.shape[-1]).view(input.shape[0],1,input.shape[-1])
                input =  torch.concat([input, indicator_col], dim=1)
                output = eval_model(input)
                one_epoch_output_log.append(output)
                val_loss += loss_fn(output, target)

        output_log.append(torch.concat(one_epoch_output_log))
        val_loss /= len(val_loader)
        val_loss_log.append( val_loss )

        msg = f"Phase2 : Epoch {epoch+1} train {config['loss_fn']} : {loss:.4f} | val {config['loss_fn']} : {val_loss:.4f}"

        # log
        utils.write_log(loss_log,  msg + "\n")

        # print(msg)
        pbar.set_description(msg)

        if val_loss <= best_acc:
            best_acc = val_loss
            torch.save(eval_model.state_dict(), result_folder+'/ckpt'+'/phase2_{}.pt'.format(config['model']))

    
    # Loss plot
    phase = 'phase2'
    path = os.path.join(config['folder_name'], phase )
    log2png(config, path=path, name=f'{phase}_loss', phase='phase', title=f'{phase}_loss',
                        train_loss_log=train_loss_log, val_loss_log=val_loss_log)
    log2gif(config, path=path, name=f'{phase}_loss', phase='phase', title=f'{phase}_loss',
                        train_loss_log=train_loss_log, val_loss_log=val_loss_log)

    # save
    torch.save(torch.tensor(val_loss_log), os.path.join(result_folder,'phase2_val_loss_log.pt'))


    output_log = torch.concat(output_log)
    phase2_results = [config, eval_model, loaders, output_log]
    return phase2_results



def last_test(base_results, phase1_results, phase2_results):

    base_config, base_model, base_loaders, base_output_log = base_results
    config1, indi_model, phase1_loaders, phase1_output_log = phase1_results
    config2, eval_model, phase2_loaders, phase2_output_log = phase2_results


    folder_name = base_config['folder_name']
    base_model.load_state_dict(torch.load(folder_name+'/base/ckpt/base_{}.pt'.format(base_config['model'])))
    indi_model.load_state_dict(torch.load(folder_name+'/phase1/ckpt/phase1_{}.pt'.format(config1['model'])))
    eval_model.load_state_dict(torch.load(folder_name+'/phase2/ckpt/phase2_{}.pt'.format(config2['model'])))

    _, val_loader, _ = phase2_loaders

    device = torch.device('cuda')
    loss_fn = getattr(nn, config2['loss_fn'])()


    # validation
    with torch.no_grad():
        base_model.eval()
        indi_model.eval()
        eval_model.eval()

        base_output = []
        eval_output = []
        Target = []
        for i, (input, target) in enumerate(val_loader):
            input, target = input.reshape(-1, input.size()[-2], input.size()[-1])[:,:-1], target.reshape(-1, target.size()[-1])   # indicator (ex. O-OCV) should be located at last channel
            input, target = input.to(device), target.to(device)
            indicator_col = indi_model(input).repeat(1, input.shape[-1]).view(input.shape[0],1,input.shape[-1])

            # base model output
            base_output.append(base_model(input))

            # eval model output
            input =  torch.concat([input, indicator_col], dim=1)
            eval_output.append(eval_model(input))

            # Target
            Target.append(target)


    base_output = torch.concat(base_output)
    eval_output = torch.concat(eval_output)
    Target = torch.concat(Target)


    # plot 1 - basic
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    base_plot = base_output.cpu().detach().numpy()
    eval_plot = eval_output.cpu().detach().numpy()
    target_plot = Target.cpu().detach().numpy()
    
    ax.plot(base_plot, linewidth=2, c='royalblue', label='base estimation')
    ax.plot(eval_plot, linewidth=2, c='crimson', label='eval estimation')
    ax.plot(target_plot, linewidth=2, c='darkviolet', label='Ground Truth')
    ax.set_xlim([0, Target.size()[0]])
    ax.set_ylim([0,1])
    # ax.set_xticks(np.linspace(0, config['epochs'], 11))
    # ax.set_yticks(np.linspace(0, 1, 11))
    # ax.axhline(y=0.03, color='lime', linestyle='--', linewidth=1)
    ax.set_title('Base / Eval / Target', fontdict={'family': 'Sans-serif', 'weight': 'bold', 'size': 14})
    ax.grid()
    ax.legend()

    fig.savefig(os.path.join(config2['folder_name'], 'base_eval_target_1'))
    plt.close()



    # plot 2 - difference
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    base_plot = base_output.cpu().detach().numpy()
    eval_plot = eval_output.cpu().detach().numpy()
    target_plot = Target.cpu().detach().numpy()
    
    ax.plot(base_plot - target_plot, linewidth=2, c='royalblue', label='base-target difference')
    ax.plot(eval_plot - target_plot, linewidth=2, c='crimson', label='eval-target difference')

    ax.set_xlim([0, Target.size()[0]])
    ax.set_ylim([-1,1])
    # ax.set_xticks(np.linspace(0, config['epochs'], 11))
    # ax.set_yticks(np.linspace(0, 1, 11))
    ax.axhline(y=0, color='lime', linestyle='solid', linewidth=1)
    ax.set_title('Difference from Ground Truth', fontdict={'family': 'Sans-serif', 'weight': 'bold', 'size': 14})
    ax.grid()
    ax.legend()

    fig.savefig(os.path.join(config2['folder_name'], 'base_eval_target_2'))
    plt.close()


    # plot 3 - output_log to gif




if __name__ == '__main__':

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=str, default='sionyu')
    parser.add_argument('--data_per_oneday', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--loss_fn', type=str, default='L1Loss')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--debug', type=str2bool, default=False)

    parser.add_argument('--base_seconds', type=int, default=25)
    parser.add_argument('--phase1_seconds', type=int, default=25)
    parser.add_argument('--phase2_seconds', type=int, default=25)

    parser.add_argument('--base_model', type=str, default='res10')
    parser.add_argument('--phase1_model', type=str, default='res10')
    parser.add_argument('--phase2_model', type=str, default='res10')

    parser.add_argument('--base_decay', type=float, default=0.98)
    parser.add_argument('--phase1_decay', type=float, default=0.98)
    parser.add_argument('--phase2_decay', type=float, default=0.98)



    parser.add_argument('--reproduce_config', type=str, default=None)


    # parser.add_argument('--split_ratio', nargs='+', help='<Required> Set flag', required=True) # python arg.py -l 1234 2345 3456 4567
    # parser.add_argument('--', type=str, default='')

    args = parser.parse_args()


    # indicator experiments (phase1,2)
    config, base_materials, phase1_materials, phase2_materials = ready(args)

    base_results = base_train(base_materials)
    phase1_results = phase1_train(phase1_materials)
    phase2_results = phase2_train(phase1_results, phase2_materials)

    # test base vs eval with their best checkpoint
    last_test(base_results, phase1_results, phase2_results)


    


