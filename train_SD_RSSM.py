import os
import numpy as np
from hashlib import md5
from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, l1_loss
from torch.distributions.kl import kl_divergence
import pytorch_lightning as pl
import wandb
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips_pytorch import LPIPS, lpips
from datasets.PixelPendulum_SD import PixelPendulumDataset_SD
from datasets.PixelPendulum_RSSM import PixelPendulumDataset
from models.SD_RSSM import Encoder, RecurrentStateSpaceModel, ObservationModel

class RSSM_PL(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # these are set according to the original paper implementation
        self.channels=1
        self.encoder = Encoder(input_channels=self.channels)
        self.rssm = RecurrentStateSpaceModel(ssm_state_dim=self.hparams.ssm_state_dim, 
                                            rnn_hidden_dim=self.hparams.rnn_hidden_dim, 
                                            pre_distr_dim=self.hparams.pre_distr_dim,
                                            min_stddev=self.hparams.min_std_dev)
        self.obs_model = ObservationModel(ssm_state_dim=self.hparams.ssm_state_dim, 
                                              rnn_hidden_dim=self.hparams.rnn_hidden_dim,
                                              output_channels=self.channels)

        #  dataset dir
        self.dataset_dir = os.environ['DATA_DIR_VIDEO']

        # signature for checkpointing/resuming
        self.signature = f'ssm_{self.hparams.ssm_state_dim}_rnn_{self.hparams.rnn_hidden_dim}_' \
            f'pre_{self.hparams.pre_distr_dim}_decs_{self.hparams.decoder_stddev}_ds_{self.hparams.dataset}_' \
            f'batch_{self.hparams.batch_size}_lr_{self.hparams.learning_rate}_nats_{self.hparams.free_nats}_' \
            f'train_input_seq_len_{self.hparams.train_input_seq_len}_'\
            f'test_input_seq_len_{self.hparams.test_input_seq_len}_'\
            f'min_std_{self.hparams.min_std_dev}_'\
            f'sd_loss_type_{self.hparams.sd_loss_type}_sd_loss_coeff_{self.hparams.sd_loss_coeff}_' \
            f'gpus_{self.hparams.gpus}_precision_{self.hparams.precision}_' \
            f'benchmark_{self.hparams.benchmark}_debug_{self.hparams.debug}' 

    def setup(self, stage):
        print('Preparing data')
        self.datasets = {}
        if self.hparams.dataset=='pendulum':
            self.test_lengths = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 
                                    170, 200, 250, 300, 350, 400, 450, 500]
            train_nframes = self.hparams.train_input_seq_len
            test_nframes = self.hparams.test_input_seq_len + np.max(self.test_lengths)
            train_fname = 'pixel_pendulum_n_10000_steps_1000_dt_0.05_angle_30-170_vel_-2.00-2.00_len_1.20-1.40_g_8.00-12.00.hd5'
            test0_fname = 'pixel_pendulum_n_1296_steps_1000_dt_0.05_angle_30-170_vel_-2.00-2.00_len_1.20-1.40_g_8.00-12.00_test.hd5'
            val_fname =   'pixel_pendulum_n_1296_steps_1000_dt_0.05_angle_30-170_vel_-2.00-2.00_len_1.20-1.40_g_8.00-12.00_val.hd5'
            test1_fname = 'pixel_pendulum_n_1296_steps_1000_dt_0.05_angle_30-170_vel_-2.00-2.00_len_1.40-1.45_g_12.00-12.50.hd5'
            test2_fname = 'pixel_pendulum_n_1296_steps_1000_dt_0.05_angle_30-170_vel_-2.00-2.00_len_1.45-1.50_g_12.50-13.00.hd5'
            self.datasets['train'] = PixelPendulumDataset_SD(os.path.join(self.dataset_dir, 'pendulum', train_fname), train_nframes)
            self.datasets['test'] = [PixelPendulumDataset(os.path.join(self.dataset_dir, 'pendulum', val_fname), test_nframes),
                                     PixelPendulumDataset(os.path.join(self.dataset_dir, 'pendulum', test0_fname), test_nframes),
                                     PixelPendulumDataset(os.path.join(self.dataset_dir, 'pendulum', test1_fname), test_nframes),
                                     PixelPendulumDataset(os.path.join(self.dataset_dir, 'pendulum', test2_fname), test_nframes)]
        else:
            raise Exception(f'Wrong dataset: {self.hparams.dataset}')

        print(f'Train Dataset length: {len(self.datasets["train"])} - Batch size {self.hparams.batch_size}')
        print(f'Test Dataset length: {len(self.datasets["test"])} - Batch size {self.hparams.batch_size_test}')

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers, shuffle=True)

    def test_dataloader(self):
        test_dloaders = []
        for dset in self.datasets['test']:
            test_dloaders.append(DataLoader(dset, 
                       batch_size=self.hparams.batch_size_test, 
                       num_workers=int(np.minimum(self.hparams.num_workers,3)), shuffle=False))
        print(f'Using {len(test_dloaders)} test dataloaders.')
        return test_dloaders

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                                lr=self.hparams.learning_rate, weight_decay=self.hparams.adam_wd,
                                eps=self.hparams.adam_eps, betas=(self.hparams.adam_b1, self.hparams.adam_b2))
        return {'optimizer': optimizer}  # 'lr_scheduler': scheduler, # 'monitor': self.hparams.monitor

    def rollout(self, batch_in, out_seq_len):
        batch_in=batch_in.to(self.device)
        with torch.no_grad():
            # All the batch_in sequence will be consumed before predicting the futures
            batch_size, input_len, c, h, w = batch_in.size()

            # embed batch with CNN
            embedded_batch = self.encoder(batch_in.contiguous().view(-1, c, h, w)).view(batch_size, input_len, -1)
            # initialize prediction placeholder
            prediction = torch.zeros(batch_size, input_len+out_seq_len, c, h, w, device=self.device)
            # initialize state and rnn hidden state with 0 vector
            ssm_state = torch.zeros(batch_size, self.hparams.ssm_state_dim, device=self.device)
            rnn_hidden = torch.zeros(batch_size, self.hparams.rnn_hidden_dim, device=self.device)

            for l in range(input_len-1):        
                # we only use the posterior while consuming input frames
                _, next_state_posterior, rnn_hidden, _, _ = \
                            self.rssm(ssm_state, rnn_hidden, embedded_batch[:,l+1])
                ssm_state = next_state_posterior.sample() 
                # creating the prediction 1 by 1 uses less memory during inference 
                # (but not during training)
                prediction[:,l+1] = self.obs_model(ssm_state, rnn_hidden)

            for l in range(input_len, input_len+out_seq_len):
                # to predict the future we only use the prior
                next_state_prior, rnn_hidden = self.rssm.prior(ssm_state, rnn_hidden)
                ssm_state = next_state_prior.sample()
                prediction[:,l] = self.obs_model(ssm_state, rnn_hidden)
        return prediction

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        num_labels = labels.size(1)
        batch = batch['trajectory']
        batch_size, seq_len, c, h, w = batch.size()
        # embed batch with CNN
        embedded_batch = self.encoder(batch.view(-1, c, h, w)).view(batch_size, seq_len, -1)

        # prepare Tensor to maintain states sequence and rnn hidden states sequence
        ssm_states = torch.zeros(batch_size, seq_len, self.hparams.ssm_state_dim, device=self.device)
        rnn_hiddens = torch.zeros(batch_size, seq_len, self.hparams.rnn_hidden_dim, device=self.device)
    
        # initialize state and rnn hidden state with 0 vector
        ssm_state = torch.zeros(batch_size, self.hparams.ssm_state_dim, device=self.device)
        rnn_hidden = torch.zeros(batch_size, self.hparams.rnn_hidden_dim, device=self.device)

        # compute state and rnn hidden sequences and kl loss
        kl_loss = 0
        sd_loss_mean = 0
        for l in range(seq_len-1):
            next_state_prior, next_state_posterior, rnn_hidden, post_mean, post_std = \
                                self.rssm(ssm_state, rnn_hidden, embedded_batch[:,l+1])
            # SD LOSS TYPE MEAN
            sd_loss_mean = sd_loss_mean + l1_loss(post_mean[:,:num_labels], labels)

            ssm_state = next_state_posterior.rsample() # sample with reparametrization trick
            ssm_states[:,l+1] = ssm_state
            rnn_hiddens[:,l+1] = rnn_hidden
            kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
            kl_loss += kl.clamp(min=self.hparams.free_nats).mean()
        kl_loss /= (seq_len - 1)

        # compute prediction sequence 
        flatten_rnn_hiddens = rnn_hiddens.view(-1, self.hparams.rnn_hidden_dim)
        flatten_ssm_states = ssm_states.view(-1, self.hparams.ssm_state_dim)
        prediction = self.obs_model(flatten_ssm_states, flatten_rnn_hiddens).view(batch_size, seq_len, c, h, w)

        # compute loss for sequence 
        pixel_loss = (1.0/self.hparams.decoder_stddev) * mse_loss(prediction[:,1:], batch[:,1:],
                                                                  reduction='none').mean([0, 1]).sum()

        if self.hparams.sd_loss_type == 'mean':
            sd_loss = sd_loss_mean
        elif self.hparams.sd_loss_type == 'stochastic':
            # SD LOSS TYPE STOCHASTIC
            labels = labels.unsqueeze(1).expand(-1, seq_len, -1)
            # torch.Size([10, 100, 200]) torch.Size([10, 100, 30]) torch.Size([10, 100, 4])
            sd_loss = l1_loss(ssm_states[:,:,:num_labels], labels)
        sd_loss = self.hparams.sd_loss_coeff * sd_loss

        # add all losses and update model parameters with gradient descent
        loss = kl_loss + pixel_loss + sd_loss

        self.log('train/loss', loss, on_epoch=True, sync_dist=True)
        self.log('train/kl_loss', kl_loss, on_epoch=True, sync_dist=True)
        self.log('train/pixel_loss', pixel_loss, on_epoch=True, sync_dist=True)
        self.log('train/sd_loss', sd_loss, on_epoch=True, sync_dist=True)
        # Use decoder std of 1, to allow comparison
        self.log('train/pixel_loss_comp', pixel_loss*self.hparams.decoder_stddev, 
                on_epoch=True, sync_dist=True)
        return loss

    def on_test_start(self):
        # init the LPIPS loss
        self.lpips_loss = LPIPS(net_type=self.hparams.lpips_backend, version='0.1').to(self.device)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        # embed batch with CNN
        batch_size, seq_len, c, h, w = batch.size()
        input_len = self.hparams.test_input_seq_len
        max_len = input_len + np.max(self.test_lengths)
        predictions = self.rollout(batch[:,:input_len], out_seq_len=max_len-input_len)

        preds=predictions[:,input_len:max_len]
        gts=batch[:,input_len:max_len]
        batch_size, time_steps, c, h, w = preds.size()

        # Fix axes for channel
        if  c == 3: # color image
            preds = preds.permute(0, 1, 4, 3, 2)
            gts = gts.permute(0, 1, 4, 3, 2)
        elif c == 1: # grayscale
            preds = preds.squeeze(2)
            gts = gts.squeeze(2) 
        else:
            raise Exception(f'Wrong number of channels: {c}')

        metrics_c = {'ssim':  [],
                     'psnr':  [],
                     'lpips': [],
                    }

        for t in range(time_steps):
            metrics_p = {k: [] for k in metrics_c.keys()}
            
            for b in range(batch_size):
                pr = preds[b, t]
                gt = gts[b, t]
                metrics_p['lpips'].append(self.lpips_loss(gt, pr).item())        
                pr=pr.cpu().numpy()
                gt=gt.cpu().numpy()
                metrics_p['ssim'].append(ssim(gt, pr))
                metrics_p['psnr'].append(psnr(gt, pr))
            
            for k, mp in metrics_p.items():
                metrics_c[k].append(mp)
            
            if t+1 in self.test_lengths:
                # Cummulative metrics
                for k, mc in metrics_c.items():
                    self.log(f'test/cumm/{k}/mean/{t+1:04d}',  np.mean(mc), sync_dist=True)
                    self.log(f'test/cumm/{k}/var/{t+1:04d}',   np.var(mc), sync_dist=True)
                # Point metrics
                for k, mp in metrics_p.items():
                    self.log(f'test/point/{k}/mean/{t+1:04d}',  np.mean(mp), sync_dist=True)
                    self.log(f'test/point/{k}/var/{t+1:04d}', np.var(mp), sync_dist=True)

        # Use unitary decoder std to allow comparison
        for ln in self.test_lengths:
            pred=predictions[:,input_len:input_len+ln].to(self.device)
            gt=batch[:,input_len:input_len+ln].to(self.device)
            pixel_loss =  mse_loss(pred, gt, reduction='none').mean([0, 1]).sum() \
                                                   / self.hparams.decoder_stddev
            self.log(f'test/pixel_loss/{ln:04d}', pixel_loss, sync_dist=True)
            self.log(f'test/pixel_loss_comp/{ln:04d}', pixel_loss*self.hparams.decoder_stddev, 
                                       sync_dist=True)
            self.log(f'test/mse/cumm/{ln:04d}', mse_loss(pred, gt) , sync_dist=True)

if __name__=='__main__':
    # For compatibility issues
    from utilities.callbacks import NewBestModelCallback
    from utilities.toolsies import seed_everything, str2bool, none_or_int, \
                                count_pars, run_cuda_diagnostics
    parser = ArgumentParser()
    # Project
    parser.add_argument('--project_name', default='dummy_rssm')
    # Model
    parser.add_argument('--model', default='RSSM')
    parser.add_argument('--ssm_state_dim', type=int, default=30, help='stochastic SSM variables')
    parser.add_argument('--rnn_hidden_dim', type=int, default=200, help='deterministic RNN hidden size')
    parser.add_argument('--pre_distr_dim', type=none_or_int, default=None, help='the intermediate pre-distribution hidden size')
    parser.add_argument('--min_std_dev', type=float, default=0.1, help='minimum prior and posterior std deviation')
    parser.add_argument('--decoder_stddev', type=float, default=1.0, help='std deviation of gaussian decoder')
    # Supervised disentanglement
    parser.add_argument('--sd_loss_type', type=str, default='mean', help='loss can be computed on posterior mean or samples(stochastic)') # epoch is appended automagically
    parser.add_argument('--sd_loss_coeff', type=float, default=1.0, help='supervised disentanglement loss coefficient')
    # Dataset 
    parser.add_argument('--dataset', default='mmnist')
    parser.add_argument('--train_input_seq_len', type=none_or_int, default=None, 
                                                 help='num of frames to use during training')
    # Training
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--batch_size_test', default=10, type=int, help='batch size for evaluation purposes')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--free_nats', type=int, default=3, help='free nats for the KL divergence')
    parser.add_argument('--adam_b1', type=float, default=0.9, help='decay rate 1')
    parser.add_argument('--adam_b2', type=float, default=0.999, help='decay rate 2')
    parser.add_argument('--adam_eps', type=float, default=1e-4, help='epsilon for numeric stability')
    parser.add_argument('--adam_wd', type=float, default=0, help='weight decay')
    # Duration
    parser.add_argument('--benchmark', default=False, type=str2bool, help='It can make training faster')
    parser.add_argument('--monitor', type=str, default='train/loss') # epoch is appended automagically
    parser.add_argument('--max_epochs', type=int, default=300, help='number of epochs to train for')
    # parser.add_argument('--max_time', type=str, default='00:23:30:00', help='time to run. cluster has 24h limit.')
    # Testing
    parser.add_argument('--test_input_seq_len', type=int, default=50, help='num of frames to consume before starting predicting during testing')
    parser.add_argument('--lpips_backend', type=str, default='alex', help='backend of the lpips loss. alex or vgg')
    # Technical 
    parser.add_argument('--use_checkpoint', default=True, type=str2bool, help='When true it resumes if possible')
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
    parser.add_argument('--precision', default=32, type=int, help='Use 16 for mixed-precision')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    # Logging
    parser.add_argument('--log_every_n_steps', type=int, default=1)
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=1)
    parser.add_argument('--fast_dev_run', type=str2bool, default=False)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--signature', type=str, default=None)
    hparams = parser.parse_args()

    if hparams.gpus>1:
        print('Multiple GPUs detected. Enabling ddp.')
        setattr(hparams, 'accelerator', 'ddp') 
        setattr(hparams, 'find_unused_parameters', False) 

    # Seed everything for reproducibility
    seed_everything(hparams.seed)
    pl.seed_everything(hparams.seed)

    if hparams.debug:
        limit_train_batches=2
        limit_test_batches=2
    else:
        limit_train_batches=1.0
        limit_test_batches=1.0

    # Model creation
    model = RSSM_PL(**vars(hparams))

    # WANDB Logger
    logger_dir = os.path.join(os.environ['WANDB_DIR'], hparams.project_name)
    os.makedirs(logger_dir, exist_ok=True)

    runid = md5(model.signature.encode('UTF-8')).hexdigest() + \
            f'_ep_{hparams.max_epochs:04d}_seed_{hparams.seed:04d}'
    print(f'Model HASH: {runid}')
    logger = pl.loggers.WandbLogger(project=hparams.project_name, 
                                    id=runid,
                                    save_dir=logger_dir,
                                    log_model=True)
    logger.log_hyperparams(vars(hparams))

    # Callbacks
    # checkpoint directory
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints/', 
                                    hparams.project_name, runid)
    os.makedirs(checkpoint_dir, exist_ok=True)

    local_checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                verbose=True, save_top_k=0, monitor=None, save_last=True)

    wandb_checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=logger.experiment.dir,
                verbose=True, save_top_k=0, monitor=None, save_last=True)

    # Resume from checkpoint if it exists
    detected_checkpoint = None
    if hparams.use_checkpoint:
        possible_checkpoint = os.path.join(checkpoint_dir,'last.ckpt')
        # load existing, or create a new checkpoint
        if os.path.isfile(possible_checkpoint):
            detected_checkpoint = possible_checkpoint
            print(f'Checkpoint detected. Loading model from: {detected_checkpoint}\n')

    newbestmodel_callback = NewBestModelCallback()
    trainer = pl.Trainer.from_argparse_args(hparams,
                                            logger=logger,
                                            callbacks=[local_checkpoint_callback,
                                                        wandb_checkpoint_callback,
                                                       newbestmodel_callback
                                                       ],
                                            resume_from_checkpoint=detected_checkpoint,
                                            terminate_on_nan=True,
                                            auto_select_gpus=True,
                                            limit_test_batches=limit_test_batches, 
                                            limit_train_batches=limit_train_batches)

    run_cuda_diagnostics(hparams.gpus)
    print('\nTraining diagnostics:')
    print('---------------------')
    print(f'Hyperparameters: {vars(hparams)}\n')
    print(f'Model signature: {model.signature}\n')
    print(f'Checkpoint dir: {checkpoint_dir}\n')
    print(f'Using previous checkpoint: {detected_checkpoint}\n')
    print(f'Total number of parameters: {count_pars(model)}\n')
    print('\nStarting training')
    print('---------------------')
    trainer.fit(model)
    trainer.test(model)