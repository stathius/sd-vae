import os
from argparse import ArgumentParser
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import wandb
import sys
import random
sys.path.append('..')
from models.MLP import MLP
from base_module import BaseDynamicsModule
from utilities.toolsies import seed_everything, str2bool
from utilities.callbacks import BestValidationCallback, TestEndCallback


class DynamicsMLP(BaseDynamicsModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.model = MLP(input_size = self.hparams.model_input_size, 
                        output_size  = self.hparams.model_output_size,
                        model_size  = self.hparams.model_hidden_size, 
                        latent_size  = self.hparams.model_latent_size, 
                        nonlinearity = self.hparams.model_nonlinearity,
                        coord_dim = self.coord_dim,
                        use_layer_norm=self.hparams.use_layer_norm)
        
        if self.hparams.use_supervision and self.hparams.sup_loss_type=='sigmoid_parametrized':
            self.w1 = torch.nn.Parameter(torch.tensor(1.0))
            self.w2 = torch.nn.Parameter(torch.tensor(1.0))
            
    def rollout(self, batch, start, rollout_size):
        trajectory = batch['trajectory']
        input_end_point = output_start_point = start + self.hparams.model_input_size
        input_trajectory = trajectory[:, start:input_end_point, :]
        output = self.model(input_trajectory)[0]

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
            output = torch.cat((output, self.model(input_trajectory)[0]), dim=1)

        return output[:, :rollout_size, :], trajectory[:, output_start_point:(output_start_point+rollout_size), :]
    
    def forward(self, batch):
        # one forward pass with the models default input output sizes
        # the starting point is randomized in here
        trajectory = batch['trajectory']
        start = self.get_start(batch, self.hparams.model_output_size)

        input_end_point = output_start_point = start + self.hparams.model_input_size
        input_trajectory = trajectory[:, start:input_end_point, :]
        target_trajectory = trajectory[:, output_start_point:(output_start_point + 
                                                self.hparams.model_output_size), :]
        output_trajectory, latents = self.model(input_trajectory)
        return output_trajectory, target_trajectory, latents

    def get_label_loss(self, batch, latents):
        labels = batch['labels']
        label_loss = self._compute_label_loss(labels, latents)
        return label_loss

    def training_step(self, train_batch, batch_idx):
        rec_loss = 0.0
        label_loss = 0.0
        for i in range(self.hparams.samples_per_batch_train):
            output_trajectory, target_trajectory, latents = self.forward(train_batch)
            rec_loss= rec_loss + self.reconstruction_loss(output_trajectory, target_trajectory)
            if (self.hparams.use_supervision):
                label_loss = label_loss + self.get_label_loss(train_batch, latents)

        rec_loss = rec_loss/self.hparams.samples_per_batch_train
        self.log('train/rec', rec_loss, prog_bar=True, on_step=False, on_epoch=True)

        if self.hparams.use_supervision:
            label_loss = label_loss/self.hparams.samples_per_batch_train
            self.log('train/label_loss', label_loss, prog_bar=True, on_step=False, on_epoch=True)
            train_loss = rec_loss + self.hparams.sup_multiplier * label_loss

        # Log longer losses
        if (batch_idx % self.hparams.log_freq) == 0:
            self.log_rec_losses(train_batch, 'train', self.val_rec_loss_sizes,
                                                on_step=False, on_epoch=True)

        return train_loss

    def validation_step(self, val_batch, batch_idx):
        for i in range(self.hparams.samples_per_batch_val):
            self.log_rec_losses(val_batch, 'val', self.val_rec_loss_sizes)

        if self.hparams.use_supervision:
            _, _, latents = self.forward(val_batch)
            label_loss = self.get_label_loss(val_batch, latents)
            self.log('val/label_loss', label_loss) 

    def test_step(self, test_batch, batch_idx, dataloader_idx=None):
        for i in range(self.hparams.samples_per_batch_test):
            self.log_rec_losses(test_batch, 'test', self.test_rec_loss_sizes)

        if self.hparams.use_supervision:
            _, _, latents = self.forward(test_batch)
            label_loss = self.get_label_loss(test_batch, latents)
            self.log('val/label_loss', label_loss)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='dummy')
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--dataset', default='var_length')
    parser.add_argument('--dataset_dt', type=float, default=0.05)
    parser.add_argument('--coordinates', default='phase_space')
    parser.add_argument('--noise_std', type=float, default=0.0)
    # L1, MSE
    parser.add_argument('--rec_loss_type', type=str, default='L1')
    parser.add_argument('--model_nonlinearity', type=str, default='relu')
    parser.add_argument('--model_hidden_size', nargs='+', type=int,  default=[400, 200])
    parser.add_argument('--model_input_size', type=int, default=10)
    parser.add_argument('--model_latent_size', type=int, default=5)
    parser.add_argument('--model_output_size', type=int, default=1)

    # SUPERVISION
    # sigmoid, sigmoid_parametrized, linear, linear_scaledled, BCE
    parser.add_argument('--use_supervision', type=str2bool, default=False)
    parser.add_argument('--sup_loss_type', type=str, default=None) 
    parser.add_argument('--sup_multiplier', type=float, default=None)

    parser.add_argument('--use_layer_norm', type=str2bool, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=16)
    parser.add_argument('--samples_per_batch_train', type=int, default=1)
    parser.add_argument('--samples_per_batch_val', type=int, default=1)
    parser.add_argument('--samples_per_batch_test', type=int, default=10)
    parser.add_argument('--use_random_start', type=str2bool)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
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
    parser.add_argument('--log_freq', type=int, default=50)
    parser.add_argument('--fast_dev_run', type=str2bool, default=False)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=100)

    hparams = parser.parse_args()
    print(hparams)
    
    seed_everything(hparams.seed)
    pl.seed_everything(hparams.seed)
    model = DynamicsMLP(**vars(hparams))
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
    trainer.test()