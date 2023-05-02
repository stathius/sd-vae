from argparse import ArgumentParser
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from base_module import BaseDynamicsModule
from models.VAE import VAE
from utilities.callbacks import BestValidationCallback, TestEndCallback
from utilities.toolsies import seed_everything, str2bool
from utilities.losses import kld_loss, geco_constraint

class DynamicsVAE(BaseDynamicsModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hparams.kld_scaling_type=='geco':
            self.geco_multiplier = 1
            self.C_ma = None
        elif self.hparams.kld_scaling_type=='beta_fixed':
            self.beta = self.hparams.beta
        elif self.hparams.kld_scaling_type=='beta_anneal':
            self.beta = self.hparams.beta_initial
            self.beta_initial = self.hparams.beta_initial
            self.beta_max = self.hparams.beta_max
            self.beta_anneal_steps = self.hparams.beta_anneal_steps
        else:
            raise Exception(f'Wrong KLD scaling method: {self.hparams.kld_scaling_type}')

        if self.hparams.model_use_extra_factors:
            self.extra_factors_dim = self.num_factors
        else:
            self.extra_factors_dim = 0

        self.model = VAE(input_dim = self.hparams.model_input_size, 
                            output_dim = self.hparams.model_output_size, 
                            latent_dim = self.hparams.model_latent_size,
                            factors_dim = self.extra_factors_dim,
                            hidden_dims = self.hparams.model_hidden_size, 
                            coord_dim = self.coord_dim,
                            nonlinearity = self.hparams.model_nonlinearity, 
                            dropout_pct = self.hparams.model_dropout_pct,
                            use_layer_norm = self.hparams.use_layer_norm
                            )

        if self.hparams.use_supervision and self.hparams.sup_loss_type=='sigmoid_parametrized':
            self.w1 = torch.nn.Parameter(torch.tensor(1.0))
            self.w2 = torch.nn.Parameter(torch.tensor(1.0))

    def rollout(self, batch, start, rollout_size):
        trajectory = batch['trajectory']
        input_end_point = output_start_point = start + self.hparams.model_input_size
        input_trajectory = trajectory[:, start:input_end_point, :]
        output, _, _, _ = self.model(input_trajectory)

        model_input_size = self.hparams.model_input_size
        model_output_size = self.hparams.model_output_size

        while output.size(1) < rollout_size: #keep rolling till we reach the required size
            #if the model output is smaller than the input use previous data
            if model_output_size < model_input_size:
                keep_from_input = model_input_size - model_output_size
                input_trajectory = torch.cat((input_trajectory[:, -keep_from_input:, :], 
                                                output[:, -model_output_size:,:]), dim=1)
            else:
                input_trajectory = output[:, -model_input_size:, :]
            new_output = self.model(input_trajectory)[0]
            output = torch.cat((output, new_output), dim=1)

        return output[:, :rollout_size, :], trajectory[:, output_start_point:(output_start_point + 
                                                                            rollout_size), :]

    def forward(self, batch):
        # one forward pass with the models default input output sizes
        # the starting point is randomized in here
        trajectory = batch['trajectory']
        start = self.get_start(batch, self.hparams.model_output_size)

        input_end_point = output_start_point = start + self.hparams.model_input_size
        input_trajectory = trajectory[:, start:input_end_point, :]
        target_trajectory = trajectory[:, output_start_point:(output_start_point + 
                                                self.hparams.model_output_size), :]
        output_trajectory, mu, logvar, factors = self.model(input_trajectory)
        return output_trajectory, target_trajectory, mu, logvar, factors

    def get_normalized_labels(self, labels):
        # make lengths 0-1
        labels_norm = {}
        for k,v in labels.items():
            labels_norm[k] = (v-self.labels_min[k])/(self.labels_max[k] - self.labels_min[k] + 1e-6)
        return labels_norm

    def get_geco_C(self, rec_loss):
        tol = self.hparams.geco_tolerance
        alpha = self.hparams.geco_alpha

        C = geco_constraint(rec_loss, tol)
        C_curr = C.detach() # keep track for logging
        if self.C_ma is None:
            self.C_ma = C.detach()
        else:
            self.C_ma = alpha * self.C_ma + (1 - alpha) * C.detach()
        C = C + (self.C_ma - C.detach()) 

        return C, C_curr

    def update_geco_multiplier(self, C):
        # clamping the langrange multiplier to avoid inf values
        speed = self.hparams.geco_speed
        clipping = self.hparams.geco_clipping
        self.geco_multiplier = self.geco_multiplier * torch.exp(speed * C.detach())
        self.geco_multiplier = torch.clamp(self.geco_multiplier, 1.0/clipping, clipping)

    def update_label_loss_multiplier(self):
        pass

    def get_label_loss(self, batch, mu, factors):
        labels = batch['labels']
        if (self.hparams.model_use_extra_factors):
            # use the extra latents for supervision
            assert (factors is not None)
            label_loss = self._compute_label_loss(labels, factors)
        else: 
            # use the regular latents for supervision
            label_loss = self._compute_label_loss(labels, mu)
        return label_loss

    def training_step(self, train_batch, batch_idx):
        rec_loss = 0.0
        kld = 0.0
        label_loss = 0.0
        for i in range(self.hparams.samples_per_batch_train):
            output_trajectory, target_trajectory, mu, logvar, factors = self.forward(train_batch)
            rec_loss = rec_loss + self.reconstruction_loss(output_trajectory, target_trajectory) 
            # normalize kld over output dimensions 
            loss_normalizer = hparams.model_output_size * self.coord_dim
            kld = kld + kld_loss(mu, logvar) / loss_normalizer

            if (self.hparams.use_supervision):
                label_loss = label_loss + self.get_label_loss(train_batch, mu, factors)

        rec_loss = rec_loss / self.hparams.samples_per_batch_train
        kld = kld / self.hparams.samples_per_batch_train
        self.log('train/rec', rec_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train/kld', kld, prog_bar=True, on_step=False, on_epoch=True)
        if self.hparams.use_supervision:
            label_loss = label_loss/self.hparams.samples_per_batch_train
            self.log('train/label_loss', label_loss, prog_bar=True, on_step=False, on_epoch=True)
            rec_loss = rec_loss + self.hparams.sup_multiplier * label_loss

        if self.hparams.kld_scaling_type=='geco':
            C, C_curr = self.get_geco_C(rec_loss)
            train_loss = self.geco_multiplier * C + kld
            self.log('train/geco/C', C_curr, prog_bar=False, on_step=False, on_epoch=True)
            self.log('train/geco/C_ma', self.C_ma.detach(), prog_bar=False, on_step=False, on_epoch=True)
            self.log('train/geco/geco_multiplier', self.geco_multiplier, prog_bar=True, on_step=False, on_epoch=True)
            self.update_geco_multiplier(C)
        elif self.hparams.kld_scaling_type=='beta_fixed':
            train_loss = rec_loss + self.beta * kld
            self.log('train/beta', self.beta, on_step=False, on_epoch=True)
        elif self.hparams.kld_scaling_type=='beta_anneal':
            train_loss = rec_loss + self.beta * kld
            new_beta = self.beta_initial + (self.beta_max - self.beta_initial)\
                                        *(self.global_step+1)/(self.beta_anneal_steps+1)
            self.beta = np.minimum(self.beta_max, new_beta)
            self.log('train/beta', self.beta, on_step=False, on_epoch=True)

        self.log('train/loss', train_loss, prog_bar=False, on_step=False, on_epoch=True)
        
        if (batch_idx % self.hparams.log_freq) == 0:
            self.log_rec_losses(train_batch, 'train', self.val_rec_loss_sizes,
                                                on_step=False, on_epoch=True)
            
            if self.hparams.debug:
                self.log_histogram("debug/mu", mu)
                self.log_histogram("debug/logvar", logvar)

        return train_loss

    def validation_step(self, val_batch, batch_idx): #dataloader_idx
        for i in range(self.hparams.samples_per_batch_val):
            self.log_rec_losses(val_batch, 'val', self.val_rec_loss_sizes)

        if self.hparams.use_supervision:
            _, _, mu, _, factors = self.forward(val_batch)
            label_loss = self.get_label_loss(val_batch, mu, factors)
            self.log('val/label_loss', label_loss)         

    def test_step(self, test_batch, batch_idx, dataloader_idx=None):
        for i in range(self.hparams.samples_per_batch_test):
            self.log_rec_losses(test_batch, 'test', self.test_rec_loss_sizes)

        if self.hparams.use_supervision:
            _, _, mu, _, factors = self.forward(test_batch)
            label_loss = self.get_label_loss(test_batch, mu, factors)
            self.log('test/label_loss', label_loss)         

if __name__ == '__main__':

    # parametrize the network
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='dummy')
    parser.add_argument('--model', default='vae')

    parser.add_argument('--dataset', default='var_length')
    parser.add_argument('--dataset_dt', type=float, default=0.05)
    # phase_space, cartesian
    parser.add_argument('--coordinates', default='phase_space')
    parser.add_argument('--noise_std', type=float, default=None)
    # L1, MSE
    parser.add_argument('--rec_loss_type', type=str, default='L1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=16)
    parser.add_argument('--samples_per_batch_train', type=int, default=1)
    parser.add_argument('--samples_per_batch_val', type=int, default=1)
    parser.add_argument('--samples_per_batch_test', type=int, default=10)
    parser.add_argument('--use_random_start', type=str2bool)

    parser.add_argument('--model_nonlinearity', type=str, default='leaky')
    parser.add_argument('--model_input_size', type=int, default=10)
    parser.add_argument('--model_hidden_size', nargs='+', type=int,  default=[500, 100])
    parser.add_argument('--model_use_extra_factors', type=str2bool, default=False)
    parser.add_argument('--model_latent_size', type=int, default=8)
    parser.add_argument('--model_output_size', type=int, default=1)
    parser.add_argument('--model_dropout_pct', type=float, default=0.0)
    parser.add_argument('--use_layer_norm', type=str2bool, default=None)
    # BETA
    parser.add_argument('--kld_scaling_type', type=str, default='beta')
    # beta_fixed
    parser.add_argument('--beta', type=float, default=None)
    # beta_anneal
    parser.add_argument('--beta_initial', type=float, default=None) #
    parser.add_argument('--beta_max', type=float, default=None)
    parser.add_argument('--beta_anneal_steps', type=float, default=None)
    # geco
    parser.add_argument('--geco_tolerance', type=float, default=None)
    parser.add_argument('--geco_alpha', type=float, default=None)
    parser.add_argument('--geco_speed', type=float, default=None)
    parser.add_argument('--geco_clipping', type=float, default=None)

    # SUPERVISION
    # sigmoid, sigmoid_parametrized, linear, linear_scaledled, BCE
    parser.add_argument('--use_supervision', type=str2bool, default=False)
    parser.add_argument('--sup_loss_type', type=str, default=None) 
    parser.add_argument('--sup_multiplier', type=float, default=None)
    parser.add_argument('--partition_latents', type=str2bool, default=False)

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--scheduler_patience', type=int, default=20)
    parser.add_argument('--scheduler_factor', type=float, default=0.3)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-7)
    parser.add_argument('--scheduler_threshold', type=float, default=1e-5)

    parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--monitor', type=str, default='val/rec/1')
    parser.add_argument('--early_stopping_patience', type=int, default=70)

    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--time', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_wandb', type=str2bool, default=False)
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--fast_dev_run', type=str2bool, default=False)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=100)

    hparams = parser.parse_args()
    print(hparams)

    seed_everything(hparams.seed)
    pl.seed_everything(hparams.seed)
    model = DynamicsVAE(**vars(hparams))
    print(model.model)

    if hparams.use_wandb:
        save_dir = os.path.join(os.environ['WANDB_DIR'], hparams.project_name)
        os.makedirs(save_dir, exist_ok=True)
        logger = pl.loggers.WandbLogger(project=hparams.project_name, save_dir=save_dir)
        logger.log_hyperparams(vars(hparams))
        if hparams.debug:
            logger.watch(model, log='all', log_freq=hparams.log_freq)
        checkpoint_dir = os.path.join(logger.experiment.dir, 'checkpoints/')
    else:
        # log_dir = os.path.join(os.environ['EXP_DIR'], 'tensorboard')
        log_dir = './tensorboard/'
        print(f'Using tensorboard from {log_dir}')
        os.makedirs(os.path.join(log_dir, hparams.project_name), exist_ok=True)
        experiment_name = f'in_{hparams.model_input_size}_out_{hparams.model_output_size}'
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
                save_top_k=1, 
                verbose=True, 
                mode='min',
                save_last=False)

    best_validation_callback = BestValidationCallback(hparams.monitor, hparams.use_wandb)
    test_end_callback = TestEndCallback(hparams.use_wandb)

    trainer = pl.Trainer.from_argparse_args(hparams, logger=logger,
                log_every_n_steps=1,
                gradient_clip_val=hparams.gradient_clip_val,
                callbacks=[ checkpoint_callback,
                            early_stop_callback, 
                            lr_monitor_callback, 
                            best_validation_callback,
                            test_end_callback
                        ],
                deterministic=True,
                progress_bar_refresh_rate=hparams.progress_bar_refresh_rate
                )

    trainer.fit(model)
    trainer.test()