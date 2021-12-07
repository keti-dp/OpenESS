import os, sys, time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tqdm import tqdm
import numpy as np
import pandas as pd
import pytorch_model_summary

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import get_nasa_dataset_loader
from utils.options import parse_args

from models.rnn import SimpleLSTM, SimpleRNN



# Base trainer
class BaseTrainer:
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda') if options.device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
        
        self.set_type = 'test' if options.test_only else 'train'
        self.max_train_time = np.inf if options.max_train_time == None or options.max_train_time < 1\
                              else options.max_train_time
        
        """
        Required values
        self.model_name
        self.model
        self.optimizer
        self.criterion
        self.train_data_loader
        self.valid_data_loader
        self.trained_epoch if options.continue_training != None
        """
        
    def train(self):
        log_dir = 'logs/' + time.strftime('%y%m%d%H%M%S_', time.localtime(time.time())) + self.model_name if self.options.continue_training == None else self.options.continue_training
        summary_writer = SummaryWriter(log_dir)
        model_temp_dir = log_dir + '/temp_state'
        
        if self.options.continue_training == None:
            if not os.path.exists(model_temp_dir):
                os.mkdir(model_temp_dir)

            if self.options.comment != None:
                with open(log_dir + '/comments.txt', 'w') as f:
                    split = self.options.comment.split(';')
                    for string in split:
                        f.write(string + '\n')

            torch.save(self.model, log_dir + '/model.pt')
            torch.save(self.options, log_dir + '/options.pt')
        
        summary_count = 1
        summary_logged_step = 0
        summary_steps = self.options.summary_steps
        
        resume_training = True
        end_time = self.max_train_time + time.time()
        
        initial_epoch = 1 if self.options.continue_training == None else self.trained_epoch + 1
        
        for epoch in tqdm(range(initial_epoch, self.options.num_epochs+1), desc='Training models'):
            self.model.train()
            for batch in tqdm(self.train_data_loader, desc='Epoch {}'.format(epoch)):
                if time.time() < end_time:
                    if summary_count < summary_steps:
                        loss = self.train_step(batch)
                        if loss != None:
                            summary_count += 1

                    else:
                        loss = self.train_step(batch)
                        if loss != None:
                            summary_logged_step += summary_steps
                            self.write_train_log(summary_writer, summary_logged_step, loss)
                            tqdm.write('Training summary logged; training loss : {}'.format(loss))
                            summary_count = 1

                else:
                    tqdm.write('Timeout reached')
                    resume_training = False
                    break

            if resume_training:
                self.model.eval()
                valid_losses = []
                for batch in self.valid_data_loader:
                    _loss = self.validate_step(batch)
                    if _loss != None:
                        valid_losses.append(_loss)
                loss = self.write_validate_log(summary_writer, epoch, valid_losses)
                self.save_checkpoint(model_temp_dir, epoch)
                loss_str = 'Validation implemented; '
                for key, value in loss.items():
                    loss_str = loss_str + key + ' : {}, '.format(value)
                tqdm.write(loss_str[:-2])
            else:
                break
                    
        self.save_checkpoint(log_dir)
                
        
    def save_checkpoint(self, log_dir, epoch=None):
        if epoch == None:
            torch.save(self.model.state_dict(), log_dir + '/state_dict.pt')
        else:
            torch.save(self.model.state_dict(), log_dir + '/state_dict_epoch-{}.pt'.format(epoch))
            
    def train_step(self, batch):
        raise NotImplementedError('"train_step" function is not provided.')
        
    def validate_step(self, batch):
        raise NotImplementedError('"validate_step" function is not provided.')
        
    def write_train_log(self, writer, step, loss):
        raise NotImplementedError('"write_train_log" function is not provided.')
        
    def write_validate_log(self, writer, epoch, losses):
        raise NotImplementedError('"write_trian_log" function is not provided.')
        
        
        
# LSTM using NASA dataset trainer
class NasaRnnTrainer(BaseTrainer):
    def __init__(self, options):
        super(NasaRnnTrainer, self).__init__(options)
        self.model_name = options.model
        columns = options.columns.split(',')
        
        if self.options.continue_training != None:
            self.model = torch.load(os.path.join(self.options.continue_training, 'model.pt')).to(self.device)
            self.trained_epoch = 0
            for file_name in os.listdir(os.path.join(self.options.continue_training, 'temp_state/')):
                if 'state_dict_epoch' in file_name:
                    self.trained_epoch += 1
            self.model.load_state_dict(torch.load(os.path.join(self.options.continue_training, 'temp_state/state_dict_epoch-{}.pt'.format(self.trained_epoch))))
            
        else:
            rnn_input_size = len(columns) if options.rnn_input_size == None else options.rnn_input_size
            if self.model_name == 'nasa_lstm':
                self.model = SimpleLSTM(rnn_input_size, options.rnn_hidden_size, options.rnn_n_layer, options.use_aggregation).to(self.device)
            elif self.model_name == 'nasa_rnn':
                self.model = SimpleRNN(rnn_input_size, options.rnn_hidden_size, options.rnn_n_layer, options.use_aggregation).to(self.device)
            
        if options.train_set != None:
            self.set_type = options.train_set
        
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=options.lr, weight_decay=1e-4)
        
        self.criterion = nn.MSELoss().to(self.device)
        
        self.train_data_loader = get_nasa_dataset_loader(set_type=self.set_type,
                                                         target=options.target,
                                                         columns=columns,
                                                         batch_size=options.batch_size,
                                                         sliding_window=options.sliding_window,
                                                         shuffle=True,
                                                         pin_memory=True,
                                                         drop_last=True)
         
        self.valid_data_loader = get_nasa_dataset_loader(set_type='valid',
                                                         target=options.target,
                                                         columns=columns,
                                                         batch_size=options.batch_size,
                                                         sliding_window=options.sliding_window,
                                                         shuffle=False,
                                                         pin_memory=True,
                                                         drop_last=False)
         
        
        self.valid_data_len = len(self.valid_data_loader.sampler)
           
        
    def train_step(self, batch):
        data = batch['data'].to(self.device)
        data = data.transpose(0, 1).contiguous()  # Transpose batch size and sliding window.
        
        gt_target = batch['target'].to(self.device)
        pred_target = self.model(data)
        
        loss = self.options.soh_loss_weight * self.criterion(pred_target.squeeze(), gt_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            
        self.optimizer.step()
        
        return loss.item()
    
    
    def validate_step(self, batch):
        with torch.no_grad():
            data = batch['data'].to(self.device)
            data = data.transpose(0, 1).contiguous()  # Transpose batch size and sliding window.

            gt_target = batch['target'].to(self.device)
            pred_target = self.model(data)

            loss = self.options.soh_loss_weight * self.criterion(pred_target.squeeze(), gt_target)
            n_batch = gt_target.shape[0]
            
            losses = {}
            losses[self.options.target + ' loss'] = loss.item() * n_batch
            
        return losses
        
        
    def write_train_log(self, writer, step, loss):
        writer.add_scalar('Loss/Train', loss, step)
        
        
    def write_validate_log(self, writer, epoch, losses):
        display = {}
        for key in losses[0].keys():
            loss = np.sum([l[key] for l in losses]) / self.valid_data_len
            writer.add_scalar('Loss/Valid : ' + key, loss, epoch)
            display[key] = loss
        return display
    
    
    

# Test the trained model.
def test_model(model_dir, epoch, log_history, options):
    device = torch.device('cuda') if options.device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')

    model = torch.load(os.path.join(model_dir, 'model.pt')).to(device)
    if epoch == None:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'state_dict.pt')))
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'temp_state/state_dict_epoch-{}.pt'.format(epoch))))
    model.eval()
    
    columns = options.columns.split(',')
    rnn_input_size = len(columns) if options.rnn_input_size == None else options.rnn_input_size
    
    print(pytorch_model_summary.summary(model, torch.zeros(options.sliding_window, options.batch_size, rnn_input_size, device=device), show_input=True))

    data_loader = get_nasa_dataset_loader(set_type='valid',
                                          target=options.target,
                                          columns=columns,
                                          batch_size=options.batch_size,
                                          sliding_window=options.sliding_window,
                                          shuffle=False,
                                          pin_memory=True,
                                          drop_last=False)
    criterion = nn.MSELoss().to(device)
    
    mse_loss = []
    data_history = {}
    cycle_history = []
    target_history = []

    for i in range(10):
        pred_history = []
        
        for batch in tqdm(data_loader, desc='Test Dataset Loading'):
            with torch.no_grad():
                data = batch['data'].to(device)
                data = data.transpose(0, 1).contiguous()  # Transpose batch size and sliding window.

                gt_target = batch['target'].to(device)
                pred_target = model(data)

                loss = criterion(pred_target.squeeze(), gt_target)
                n_batch = gt_target.shape[0]

                mse_loss.append(loss.item() * n_batch)
                
                if log_history:
                    if cycle_history != None:
                        cycle_history.extend(batch['cycle'])
                        target_history.extend(gt_target.cpu().numpy())
                    pred_history.extend(pred_target.squeeze().cpu().numpy())
                    
        if log_history:
            if cycle_history != None:
                data_history['cycle'] = cycle_history
                data_history['target'] = target_history
                cycle_history = None
            data_history['estimation_{}'.format(i+1)] = pred_history

    if log_history:
        pd.DataFrame(data_history).to_csv(os.path.join(model_dir, 'test_history.csv'))
        
    print(options.target + " loss :", np.sqrt(np.sum(mse_loss) / (10 * len(data_loader.sampler))))

    
    
if __name__ == "__main__":
    options = parse_args()
    if options.continue_training != None:
        log_dir = options.continue_training
        options = torch.load(os.path.join(log_dir, 'options.pt'))
        options.continue_training = log_dir
    
    if options.model == 'nasa_lstm' or options.model == 'nasa_rnn':
        if options.test_model != None:
            model_dir = options.test_model
            epoch = options.test_epoch
            log_history = options.log_test_history
            options = torch.load(os.path.join(model_dir, 'options.pt'))
            test_model(model_dir, epoch, log_history, options)
        else:
            trainer = NasaRnnTrainer(options)
            trainer.train()