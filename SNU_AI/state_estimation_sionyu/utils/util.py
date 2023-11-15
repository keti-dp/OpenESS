import os
import torch
import random
import numpy as np
import argparse
import yaml
import shutil

import matplotlib.pyplot as plt

from PIL import Image
from IPython.display import Image as Img
from IPython.display import display

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# import models
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def config_update(config, args):
    for key, value in vars(args).items():
        config[key] = value

def config_update_indicator(config, args):
    # reproduce case
    # argparse is meaningless
    output_dir = config['common']['output_dir']
    if args.reproduce_config:
        print('Load the reproduce_config. Only the \'output_dir\' will survive and others will be overloaded')
        with open(args.reproduce_config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['common']['output_dir'] = output_dir
        return config

    # not reproduce.
    else:
        args_dict = vars(args)
        for key, value in args_dict.items():
            if key.startswith('base'):
                k = key.split('_')[-1]
                config['base'][k] = value
            elif key.startswith('phase1'):
                k = key.split('_')[-1]
                config['phase1'][k] = value
            elif key.startswith('phase2'):
                k = key.split('_')[-1]
                config['phase2'][k] = value
            else:
                config['common'][key] = value
        return config




def log2png(config, path, name, phase,
               title, xlim=None, ylim=None, hline=0.03, for_gif=False, *colors, **logs):
    # colors should be list of color names of plt
    # each logs maybe form of [tensor, tensor, tensor, ...,  tensor]
        
    os.makedirs(path, exist_ok=True)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)

    if not colors:
        colors = ['royalblue', 'crimson', 'darkviolet']

    # plot **logs
    for i, (log_name, log) in enumerate(logs.items()):
        ax.plot([tensor.cpu().detach().numpy() for tensor in log], linewidth=2, c=colors[i], label=log_name)


    if not xlim :
        xlim = [0, config['epochs']]
    if not ylim :
        ylim = [0,1]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 11))
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 11))
    if hline:
        ax.axhline(y=hline, color='lime', linestyle='--', linewidth=1, label='Goal')
        ax.set_yticks(np.append(np.linspace(ylim[0], ylim[1], 11), hline))

    ax.set_title(title, fontdict={'family': 'Sans-serif', 'weight': 'bold', 'size': 14})
    ax.grid()
    ax.legend()


    fig.savefig( os.path.join(path, name+'.png') )

    # if for_gif:
    #     fig.savefig( os.path.join(path, 'for_gif', name+'.png') )
    # else :
    #     fig.savefig( os.path.join(path, name+'.png') )
    plt.close()


def log2gif(config, path, name, phase,
               title, xlim=[0,200], ylim=[0,1], hline=0.03, for_gif=True, rm_after=True, *colors, **logs):
    
    for_gif_path = os.path.join(path, 'for_gif')
    png_name = 'Epoch{}'
    os.makedirs( for_gif_path, exist_ok=True )


    for i in range(config['epochs']) :
        logs_until_epoch = {}
        for key, val in logs.items():
            logs_until_epoch[key] = val[:i]

        log2png(config=config, path=for_gif_path, name=png_name.format(i), phase=phase,
                            title=title, hline=hline, for_gif=for_gif, *colors, **logs_until_epoch)


    png_files = []
    for i in range(config['epochs']) :
        png_files.append( os.path.join(for_gif_path, png_name.format(i))+'.png' )

    pngs = [Image.open(x) for x in png_files]
    start = pngs[0]
    start.save( os.path.join(path, f'{name}.gif')
                    , save_all=True, append_images=pngs[1:], loop=20, duration=100)
    
    
    if rm_after:
        shutil.rmtree(for_gif_path)


def plot_multi(data, cols=None, spacing=.1, **kwargs):
    # use only dataframe
    
    from pandas.plotting._matplotlib.style import get_standard_colors

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = get_standard_colors(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])
        
        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax