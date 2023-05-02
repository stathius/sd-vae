import os
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
import sys
import numpy as np
sys.path.append('..')
from datasets.Pendulum import PendulumDataset
from datasets.LotkaVolterra import LotkaVolterraDataset
from datasets.NBody import NBodyDataset
from datasets.PixelPendulum import PixelPendulumDataset
from utilities.losses import *

class BaseDynamicsModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.rec_loss_type == 'MSE':
            self.reconstruction_loss = mse_loss
        elif self.hparams.rec_loss_type == 'L1':
            self.reconstruction_loss = l1_loss
        elif self.hparams.rec_loss_type == 'BCE_LOGITS':
            self.reconstruction_loss = bce_with_logits_loss
        elif self.hparams.rec_loss_type == 'CNN_MSE':
            self.reconstruction_loss = cnn_vae_mse_loss
        else:
            raise Exception(f'Wrong loss type {self.hparams.rec_loss_type}')

        self.val_rec_loss_sizes = [1, 5, 10, 20]
        self.test_rec_loss_sizes = [1, 5, 10, 20, 50, 100, 200]

        if self.hparams.model_output_size not in self.val_rec_loss_sizes:
            self.val_rec_loss_sizes.append(self.hparams.model_output_size)
        self.train_ind = None


    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.hparams.batch_size,
                         num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.hparams.batch_size_val, 
                    num_workers=self.hparams.num_workers, shuffle=False)

    def test_dataloader(self):
        test_dloaders = []
        for dset in self.datasets['test']:
            test_dloaders.append(DataLoader(dset, batch_size=self.hparams.batch_size_val, 
                    num_workers=self.hparams.num_workers, shuffle=False))
        return test_dloaders

    def setup(self, stage):
        # prepare a train and validation batches
        self.batch_sample = {}
        self.batch_sample['train'] = next(iter(self.train_dataloader()))
        self.batch_sample['val'] = next(iter(self.val_dataloader()))

        if self.hparams.dataset=='pendulum_var_length':
            self.batch_sample['test_short'] = next(iter(self.test_dataloader()[1]))
            self.batch_sample['test_long'] = next(iter(self.test_dataloader()[2]))
        if self.hparams.dataset=='pendulum-2':
            self.batch_sample['test_out_1'] = next(iter(self.test_dataloader()[3]))
            self.batch_sample['test_out_2'] = next(iter(self.test_dataloader()[4]))
        elif self.hparams.dataset=='3body-2':
            self.batch_sample['test_easy'] = next(iter(self.test_dataloader()[1]))
            self.batch_sample['test_hard'] = next(iter(self.test_dataloader()[2]))
        else:
            # just get a sample from the first test dataset
            # index is 1 because 0 is the val dataset
            self.batch_sample['test'] = next(iter(self.test_dataloader()[1]))

        self.labels_min, self.labels_max = self.datasets['train'].get_labels_min_max()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay)
        # try CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                mode='min', 
                                                factor=self.hparams.scheduler_factor,
                                                patience=self.hparams.scheduler_patience, 
                                                verbose=True,
                                                threshold=self.hparams.scheduler_threshold,
                                                min_lr=self.hparams.scheduler_min_lr)
        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler,
                'monitor': self.hparams.monitor}

    def log_histogram(self, name, params):
        self.logger.experiment.log({name: wandb.Histogram(params.detach())}, step=self.global_step)

    def log_image(self, name, plt_image):
            self.logger.experiment.log({name: wandb.Image(plt_image)}, step=self.global_step)

    def log_rec_losses(self, batch, stage, rec_loss_sizes, on_epoch=True, on_step=False):
        # reconstruction losses for longer trajectories.
        max_rollout = np.max(rec_loss_sizes) 
        start = self.get_start(batch, rec_loss_sizes)

        output, target = self.rollout(batch, start=start, rollout_size=max_rollout)
        output = output.to(self.device)
        target = target.to(self.device)
        for step in rec_loss_sizes:
            rec_loss = self.reconstruction_loss(output[:, :step], target[:, :step])
            self.log(f'{stage}/rec/cumm/{step:04d}', rec_loss, on_step=on_step, on_epoch=on_epoch)

            rec_loss = self.reconstruction_loss(output[:, (step-1):step], target[:, (step-1):step])
            self.log(f'{stage}/rec/{step:04d}', rec_loss, on_step=on_step, on_epoch=on_epoch)


    def _on_after_backward(self): # remove preceding underscore to enable
    # used to log parameters grads. pl/wandb has gradient norm logging which is easier/lighter
        if self.hparams.debug and ((self.global_step % self.hparams.log_freq) == 0):
            for name, params in self.named_parameters():
                self.log_histogram(name, params)
                self.log_histogram(f'grads/{name}', params.grad.data)

    def get_start(self, batch, rec_loss_sizes):
        if self.hparams.use_random_start==True:
            length = batch['trajectory'].size(1)
            max_rollout = np.max(rec_loss_sizes) 
            max_start = length - self.hparams.model_input_size - max_rollout
            start = np.random.choice(range(max_start))
        else:
            start = 0
        return start

    def _compute_label_loss(self, labels, mu):
        # loss for labels
        labels_min, labels_max = self.labels_min, self.labels_max

        if self.hparams.sup_loss_type == 'sigmoid':
            pred_scaled = torch.sigmoid(mu[:, :self.num_factors]) \
                            * (labels_max - labels_min) + labels_min
            label_loss = F.l1_loss(pred_scaled, labels)
        elif self.hparams.sup_loss_type == 'sigmoid_parametrized':
            pred_scaled = self.w1 * torch.sigmoid(self.w2 * mu[:, :self.num_factors]) * \
                        (labels_max - labels_min) + labels_min
            label_loss = F.l1_loss(pred_scaled, labels)
            self.log('train/sup/w2', self.w2, prog_bar=False, on_step=False, on_epoch=True)
            self.log('train/sup/w1', self.w1, prog_bar=False, on_step=False, on_epoch=True)
        elif self.hparams.sup_loss_type == 'linear':
            pred_scaled = mu[:, :self.num_factors]
            label_loss = F.l1_loss(pred_scaled, labels)
        elif self.hparams.sup_loss_type == 'linear_scaled':
            pred_scaled = mu[:, :self.num_factors] * \
                            (labels_max - labels_min) + labels_min
            label_loss = F.l1_loss(pred_scaled, labels)
        elif self.hparams.sup_loss_type == 'BCE':
            labels_norm = (labels-labels_min)/(labels_max - labels_min + 1e-6)
            BCE = torch.nn.BCEWithLogitsLoss(reduction='mean')
            label_loss = BCE(mu[:, :self.num_factors], labels_norm)
        else:
            raise Warning('Wrong supervised loss type: ', self.hparams.sup_loss_type)
        return label_loss