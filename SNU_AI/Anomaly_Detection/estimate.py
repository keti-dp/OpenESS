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
import argparse
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from utils.dataset import ESS_dataset
import utils.config as config

# Exponential weighted moving average
def ewma(series, weighting_factor=0.9):
    current_factor = 1 - weighting_factor
    _ewma = series.copy()
    for i in range(1, len(_ewma)):
        _ewma[i] = _ewma[i-1] * weighting_factor + _ewma[i] * current_factor
    return _ewma


# Get anomaly sequences.
def anomaly_sequence(label):
    anomaly_args = np.argwhere(label).flatten()  # Indices for abnormal points.
    
    # Terms between abnormal invervals
    terms = anomaly_args[1:] - anomaly_args[:-1]
    terms = terms > 1

    # Extract anomaly sequences.
    sequence_args = np.argwhere(terms).flatten() + 1
    sequence_length = list(sequence_args[1:] - sequence_args[:-1])
    sequence_args = list(sequence_args)

    sequence_args.insert(0, 0)
    if len(sequence_args) > 1:
        sequence_length.insert(0, sequence_args[1])
    sequence_length.append(len(anomaly_args) - sequence_args[-1])

    # Get anomaly sequence arguments.
    sequence_args = anomaly_args[sequence_args]
    anomaly_label_seq = np.transpose(np.array((sequence_args, sequence_args + np.array(sequence_length))))
    return anomaly_label_seq, sequence_length


# Interval-dependent point
def interval_dependent_point(sequences, lengths):
    n_intervals = len(sequences)
    n_steps = np.sum(lengths)
    return (n_steps / n_intervals) / lengths


def ess_score(gt, pr, anomaly_rate=0.05, adjust=True, modify=False):
    # get anomaly intervals
    gt_aug = np.concatenate([np.zeros(1), gt, np.zeros(1)]).astype(np.int32)
    gt_diff = gt_aug[1:] - gt_aug[:-1]

    begin = np.where(gt_diff == 1)[0]
    end = np.where(gt_diff == -1)[0]

    intervals = np.stack([begin, end], axis=1)

    # quantile cut
    pa = pr.copy()
    q = np.quantile(pa, 1-anomaly_rate)
    pa = (pa > q).astype(np.int32)
    
    # Modified Ess
    if modify:
        gt_seq_args, gt_seq_lens = anomaly_sequence(gt)  # gt anomaly sequence args
        ind_p = interval_dependent_point(gt_seq_args, gt_seq_lens)  # interval-dependent point
        
        # Compute TP and FN.
        TP = 0
        FN = 0
        for _seq, _len, _p in zip(gt_seq_args, gt_seq_lens, ind_p):
            n_tp = pa[_seq[0]:_seq[1]].sum()
            n_fn = _len - n_tp
            TP += n_tp * _p
            FN += n_fn * _p
            
        # Compute TN and FP.
        TN = ((1 - gt) * (1 - pa)).sum()
        FP = ((1 - gt) * pa).sum()

    else:
        # point adjustment
        if adjust:
            for s, e in intervals:
                interval = slice(s, e)
                if pa[interval].sum() > 0:
                    pa[interval] = 1

        # confusion matrix
        TP = (gt * pa).sum()
        TN = ((1 - gt) * (1 - pa)).sum()
        FP = ((1 - gt) * pa).sum()
        FN = (gt * (1 - pa)).sum()

        assert (TP + TN + FP + FN) == len(gt)

    # Compute p, r, ess.
    precision = 0 if TP+FP == 0 else TP / (TP + FP)
    recall = 0 if TP+FN == 0 else TP / (TP + FN)
    ess_score = (precision+recall)/2

    return precision, recall, ess_score

# Estimate anomaly scores.
def estimate(test_data, model, post_activation, out_dim, batch_size, window_sliding, check_count=None, device='cpu'):
    """
    Writer : parkis

    Computes anomaly scores of test data
    """
    # Estimation settings
    window_size = model.max_seq_len * model.patch_size # 512 * 90 = 46080
    assert window_size % window_sliding == 0
    
    n_column = out_dim # 1
    n_batch = batch_size # 16
    batch_sliding = n_batch * window_size # 16 * 46080 = 737280
    _batch_sliding = n_batch * window_sliding # 16 * 512 = 8192
    data_len = len(test_data)

    count = 0
    checked_index = np.inf if check_count == None else check_count
    
    # Record output values.
    last_window = data_len - window_size + 1
    output_values = torch.zeros(data_len, n_column, device=device)
    n_overlap = torch.zeros(data_len, device=device)

    with torch.no_grad():
        _first = -batch_sliding
        for first in range(0, last_window-batch_sliding+1, batch_sliding):
            for i in range(first, first+window_size, window_sliding):
                # Call mini-batch data.
                x = torch.Tensor(test_data[i:i+batch_sliding].copy()).reshape(n_batch, window_size, -1).to(device)
                
                # Evaludate and record errors.
                y = post_activation(model(x))
                output_values[i:i+batch_sliding] += y.view(-1, n_column)
                n_overlap[i:i+batch_sliding] += 1

                count += n_batch

                if count > checked_index:
                    print(count, 'windows are computed.')
                    checked_index += check_count

            _first = first

        _first += batch_sliding

        for first, last in zip(range(_first, last_window, _batch_sliding),
                                list(range(_first+_batch_sliding, last_window, _batch_sliding)) + [last_window]):
            # Call mini-batch data.
            x = []
            for i in list(range(first, last-1, window_sliding)) + [last-1]:
                x.append(torch.Tensor(test_data[i:i+window_size].copy()))

            # Reconstruct data.
            x = torch.stack(x).to(device)

            # Evaludate and record errors.
            y = post_activation(model(x))
            for i, j in enumerate(list(range(first, last-1, window_sliding)) + [last-1]):
                output_values[j:j+window_size] += y[i]
                n_overlap[j:j+window_size] += 1

            count += n_batch

            if count > checked_index:
                print(count, 'windows are computed.')
                checked_index += check_count

        # Compute mean values.
        output_values = output_values / n_overlap.unsqueeze(-1)

    return output_values

def compute(test_data, test_label, output_values, prefix, options):
    """
    Writer : parkis
    
    Compute ess scores from anomaly scores of test data

    Args:
        test_data (np.array) : data degraded from original test data which is not degraded
        test_label (np.array) : whether degradation exists or not
        output_values (np.array) : anomaly scores of test data
        prefix (str) : prefix of save file
    """
    if output_values.ndim == 2:
        output_values = output_values[:, 0]
    
    if options.smooth_scores:
        smoothed_values = ewma(output_values, options.smoothing_weight)
    
    result_file = prefix + '_evaluations.txt'
    result_file = open(result_file, 'w')
        
    # Compute ess-scores.
    ess_str = 'Modified ess-score' if options.modified_ess else 'ess-score'

    # ess Without PA
    result_file.write('<'+ess_str+' without point adjustment>\n\n')
    
    best_eval = (0, 0, 0)
    best_rate = 0
    for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
        evaluation = ess_score(test_label, output_values, rate, False, options.modified_ess)
        result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | ess-score: {evaluation[2]:.5f}\n')
        if evaluation[2] > best_eval[2]:
            best_eval = evaluation
            best_rate = rate
    result_file.write('\nBest ess-score\n')
    result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n\n\n')
    print('Best ess-score without point adjustment')
    print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n')
    
    # ess With PA
    if not options.modified_ess:
        result_file.write('<ess-score with point adjustment>\n\n')
        
        best_eval = (0, 0, 0)
        best_rate = 0
        for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
            evaluation = ess_score(test_label, output_values, rate, True)
            result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | ess-score: {evaluation[2]:.5f}\n')
            if evaluation[2] > best_eval[2]:
                best_eval = evaluation
                best_rate = rate
        result_file.write('\nBest ess-score\n')
        result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n\n\n')
        print('Best ess-score with point adjustment')
        print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n')
    
    if options.smooth_scores:
        # ess Without PA
        result_file.write('<'+ess_str+' of smoothed scores without point adjustment>\n\n')
        best_eval = (0, 0, 0)
        best_rate = 0
        for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
            evaluation = ess_score(test_label, smoothed_values, rate, False, options.modified_ess)
            result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | ess-score: {evaluation[2]:.5f}\n')
            if evaluation[2] > best_eval[2]:
                best_eval = evaluation
                best_rate = rate
        result_file.write('\nBest ess-score\n')
        result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n\n\n')
        print('Best ess-score of smoothed scores without point adjustment')
        print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n')
        
        # ess With PA
        if not options.modified_ess:
            result_file.write('<ess-score of smoothed scores with point adjustment>\n\n')
            best_eval = (0, 0, 0)
            best_rate = 0
            for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
                evaluation = ess_score(test_label, smoothed_values, rate, True)
                result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | ess-score: {evaluation[2]:.5f}\n')
                if evaluation[2] > best_eval[2]:
                    best_eval = evaluation
                    best_rate = rate
            result_file.write('\nBest ess-score\n')
            result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n\n\n')
            print('Best ess-score of smoothed scores with point adjustment')
            print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n')
    
    # Close file.
    result_file.close()

def main(args):
    options = OmegaConf.load(args.options)

    if options.model is None:
        options.model = os.path.join(options.base_folder, 'model.pt')
    else:
        options.model = os.path.join(options.base_folder, options.model)
    
    if options.state_dict is None:
        if 'lora' in options.base_folder:
            options.state_dict = os.path.join(options.base_folder, 'lora_state_dict.pt')
        else:
            options.state_dict = os.path.join(options.base_folder, 'state_dict.pt')
    else:
        options.state_dict = os.path.join(options.base_folder, options.state_dict)

    train_file = torch.load(options.state_dict, map_location='cpu')
    train_options = train_file['options']
    options = OmegaConf.merge(train_options, options)

    if options.dataset is None:
        options.dataset = train_options.dataset
    
    if options.dataset in config.TEST_DATASET:
        name = options.dataset
    else:
        name = 'other'

    save_folder = os.path.join(options.log_dir, time.strftime('%y%m%d%H%M%S_'+name, time.localtime(time.time()))+'_results')
    # save_folder = options.state_dict[:-3]+'_results'
    options.save_folder = save_folder
    os.makedirs(save_folder, exist_ok=True)

    # Load model.
    device = torch.device('cuda:{}'.format(args.gpu_id))
    model = torch.load(options.model, map_location=device)
    
    if 'lora' in options.log_dir:
        model.load_state_dict(torch.load(options.checkpoint, map_location='cpu')['model_state_dict'], strict=False) # base weight
        model.load_state_dict(train_file['model_state_dict'], strict=False) # lora weight
    else:
        model.load_state_dict(train_file['model_state_dict'])
    model.eval()

    test_dataset = ESS_dataset(options=options, seed=args.seed, is_train=False)
    if options.data.deg_num is None:
        options.data.deg_num = test_dataset.time_len // 86400
    test_data, test_label = test_dataset.get_test_data(deg_num=options.data.deg_num, save=options.save_data)

    n_column = 1
    post_activation = torch.nn.Sigmoid().to(device)
            
    # Estimate scores.
    output_values = estimate(test_data, model, post_activation, n_column, options.batch_size,
                             options.window_sliding, options.evaluate.check_count, device)
    
    # Save results.
    output_values = output_values.cpu().numpy()
    outfile = os.path.join(save_folder, name + '_results.npy')
    np.save(outfile, output_values)

    with open(f'{save_folder}/{name}_hyperparameters.yaml', 'w') as f:
        OmegaConf.save(config=options, f=f)
    
    compute(test_data, test_label, output_values, outfile[:-4], options.evaluate)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", default='./configs/test_config.yaml')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu_id", default=0, type=int)
    args = parser.parse_args()

    main(args)
