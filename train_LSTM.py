import os
from argparse import ArgumentParser
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import wandb
import sys
import random
import numpy as np
sys.path.append('..')
from models.LSTM import S2S_LSTM
from base_module import BaseDynamicsModule
from utilities.toolsies import seed_everything, str2bool
from utilities.callbacks import BestValidationCallback, TestEndCallback


class DynamicsLSTM(BaseDynamicsModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = S2S_LSTM(coord_dims=self.coord_dim,
                              hidden_size=self.hparams.model_hidden_size, 
                              num_layers=self.hparams.model_num_layers)
        
    def rollout(self, batch, start, rollout_size, refeed=True):
        trajectory = batch['trajectory'].to(self.device)
        input_end_point = output_start_point = start + self.hparams.model_input_size
        input_trajectory = trajectory[:, start:input_end_point, :]
        ground_truth = trajectory[:, output_start_point:(output_start_point+rollout_size), :]

        if refeed:
            output = self.model(input_trajectory, target_len=self.hparams.model_output_size)
            while output.size(1) < rollout_size: #keep rolling till we reach the required size
                #if the model output is smaller than the input use previous data
                if self.hparams.model_output_size < self.hparams.model_input_size:
                    keep_from_input = self.hparams.model_input_size - self.hparams.model_output_size
                    input_trajectory = torch.cat((input_trajectory[:, -keep_from_input:, :], 
                                                    output[:, -self.hparams.model_output_size:,:]), dim=1)
                else:
                    input_trajectory = output[:, -self.hparams.model_input_size:, :]
                out_one = self.model(input_trajectory, target_len=self.hparams.model_output_size)
                output = torch.cat((output, out_one), dim=1)

            return output[:, :rollout_size, :], ground_truth
        else:
            output = self.model(input_trajectory, target_len=rollout_size)
            return output, ground_truth
    
    def forward(self, batch):
        # one forward pass with the models default input output sizes
        # the starting point is randomized in here
        target_len = self.hparams.model_output_size
        trajectory = batch['trajectory']
        start = self.get_start(batch, target_len)

        input_end_point = output_start_point = start + self.hparams.model_input_size
        input = trajectory[:, start:input_end_point, :]
        target = trajectory[:, output_start_point:(output_start_point + target_len), :]

        tfr = self.hparams.teacher_forcing_ratio
        # use teacher forcing
        if (tfr>0.0) and (random.random() < tfr):
            output = self.model(input, target_len, target)
        else: # predict recursively
            output = self.model(input, target_len, None)
        return output, target

    def training_step(self, train_batch, batch_idx):
        rec_loss = 0.0
        for i in range(self.hparams.samples_per_batch_train):
            output_trajectory, target_trajectory = self.forward(train_batch)
            rec_loss= rec_loss + self.reconstruction_loss(output_trajectory, target_trajectory)

        rec_loss = rec_loss/self.hparams.samples_per_batch_train
        self.log('train/rec', rec_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Log longer losses
        with torch.no_grad():
            if (batch_idx % self.hparams.log_freq) == 0:
                self.log_rec_losses(train_batch, 'train', self.val_rec_loss_sizes)
        return rec_loss

    def training_epoch_end(self, outputs):
        if self.hparams.teacher_forcing_ratio>0.0:
            self.hparams.teacher_forcing_ratio -= self.hparams.teacher_forcing_reduction
        else:
            self.hparams.teacher_forcing_ratio=0.0
        self.log('train/teacher_forcing_ratio', self.hparams.teacher_forcing_ratio)

    def validation_step(self, val_batch, batch_idx):
        for i in range(self.hparams.samples_per_batch_val):
            self.log_rec_losses(val_batch, 'val', self.val_rec_loss_sizes)

    def test_step(self, test_batch, batch_idx, dataloader_idx=None):
        for i in range(self.hparams.samples_per_batch_test):
            self.log_rec_losses(test_batch, 'test', self.test_rec_loss_sizes)

    def log_rec_losses(self, batch, stage, rec_loss_sizes, on_epoch=True, on_step=False):
        # reconstruction losses for longer trajectories.
        max_rollout = np.max(rec_loss_sizes) 
        start = self.get_start(batch, rec_loss_sizes)

        output, target = self.rollout(batch, start=start, rollout_size=max_rollout, refeed=True)
        output = output.to(self.device)
        target = target.to(self.device)
        for step in rec_loss_sizes:
            rec_loss = self.reconstruction_loss(output[:, :step], target[:, :step])
            self.log(f'{stage}/rec/cumm/{step:04d}', rec_loss, on_step=on_step, on_epoch=on_epoch)
            rec_loss = self.reconstruction_loss(output[:, (step-1):step], target[:, (step-1):step])
            self.log(f'{stage}/rec/{step:04d}', rec_loss, on_step=on_step, on_epoch=on_epoch)

        # Also log the internal h propagation results
        output, target = self.rollout(batch, start=start, rollout_size=max_rollout, refeed=False)
        output = output.to(self.device)
        target = target.to(self.device)
        for step in rec_loss_sizes:
            rec_loss = self.reconstruction_loss(output[:, :step], target[:, :step])
            self.log(f'{stage}/rec/cumm/hprop/{step:04d}', rec_loss, on_step=on_step, on_epoch=on_epoch)
            rec_loss = self.reconstruction_loss(output[:, (step-1):step], target[:, (step-1):step])
            self.log(f'{stage}/rec/hprop/{step:04d}', rec_loss, on_step=on_step, on_epoch=on_epoch)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='dummy')
    parser.add_argument('--model', default='lstm')
    parser.add_argument('--dataset', default='pendulum-2')
    parser.add_argument('--dataset_dt', type=float, default=0.05)
    parser.add_argument('--coordinates', default='phase_space')
    parser.add_argument('--noise_std', type=float, default=0.0)
    parser.add_argument('--rec_loss_type', type=str, default='L1')

    parser.add_argument('--model_hidden_size', type=int,  default=100)
    parser.add_argument('--model_num_layers', type=int,  default=2)
    parser.add_argument('--model_input_size', type=int, default=10)
    parser.add_argument('--model_output_size', type=int, default=1)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0)
    parser.add_argument('--teacher_forcing_reduction', type=float, default=0.05)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--batch_size_val', type=int, default=16)
    parser.add_argument('--samples_per_batch_train', type=int, default=1)
    parser.add_argument('--samples_per_batch_val', type=int, default=1)
    parser.add_argument('--samples_per_batch_test', type=int, default=10)
    parser.add_argument('--use_random_start', type=str2bool)
    parser.add_argument('--model_dropout_pct', type=float, default=0.0)
    parser.add_argument('--scheduler_patience', type=int, default=20)
    parser.add_argument('--scheduler_factor', type=float, default=0.3)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-7)
    parser.add_argument('--scheduler_threshold', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--monitor', type=str, default='val/rec/0001')
    parser.add_argument('--early_stopping_patience', type=int, default=60)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--fast_dev_run', type=str2bool, default=False)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=100)

    hparams = parser.parse_args()
    print(hparams)
    
    seed_everything(hparams.seed)
    pl.seed_everything(hparams.seed)
    model = DynamicsLSTM(**vars(hparams))
    print(model)

    if hparams.use_wandb:
        save_dir = os.path.join(os.environ['WANDB_DIR'], hparams.project_name)
        os.makedirs(save_dir, exist_ok=True)
        logger = pl.loggers.WandbLogger(project=hparams.project_name, save_dir=save_dir)
        logger.log_hyperparams(vars(hparams))
        if hparams.debug:
            logger.watch(model)
        checkpoint_dir = os.path.join(logger.experiment.dir, 'checkpoints/')
    else:
        # log_dir = os.path.join(os.environ['EXP_DIR'], 'tensorboard')
        log_dir = '~/tensorboard/'
        print(f'Using tensorboard from {log_dir}')
        os.makedirs(os.path.join(log_dir, hparams.project_name), exist_ok=True)
        experiment_name = f'in_{hparams.model_input_size}_out{hparams.model_output_size}'
        logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name)
        checkpoint_dir = logger.log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f'Checkpoint dir {checkpoint_dir}')

    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    early_stop_callback = pl.callbacks.EarlyStopping(monitor=hparams.monitor, min_delta=0.00, 
                patience=hparams.early_stopping_patience, verbose=True, mode='min')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='{epoch}',
                monitor=hparams.monitor, 
                save_top_k=1,verbose=True, mode='min',
                save_last=False)
    best_validation_callback = BestValidationCallback(hparams.monitor, hparams.use_wandb)
    test_end_callback = TestEndCallback(hparams.use_wandb)
    
    trainer = pl.Trainer.from_argparse_args(hparams, logger=logger,
                log_every_n_steps=1,
                callbacks=[checkpoint_callback,
                            early_stop_callback, 
                            lr_monitor_callback, 
                            best_validation_callback,
                            test_end_callback
                           ],
                deterministic=True,
                progress_bar_refresh_rate=hparams.progress_bar_refresh_rate
                )
    trainer.fit(model)
    if not hparams.fast_dev_run:
        trainer.test()