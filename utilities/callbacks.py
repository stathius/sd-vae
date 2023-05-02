import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import figaspect
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import figaspect
import wandb 
import time 
from pathlib import Path

def create_video(pl_module, batch):
    batch = batch.to(pl_module.device)
    batch_size, seq_len, c, h, w = batch.shape
    input_seq_len = pl_module.hparams.test_input_seq_len
    out_seq_len = seq_len - input_seq_len

    sidx = np.random.randint(batch.shape[0])
    prediction = pl_module.rollout(batch[sidx:(sidx+1),:input_seq_len], out_seq_len)

    t_in_max = pl_module.hparams.test_input_seq_len
    t_in= [0, t_in_max//3-1, t_in_max//2-1, int(t_in_max//1.5)-1, t_in_max-1]  
    t_out = (np.array(pl_module.test_lengths) - 1)+input_seq_len

    nframes_in = len(t_in)
    nframes_out = len(t_out)

    all_images = torch.cat([batch[sidx:(sidx+1),:], 
                            prediction, 
                            torch.pow(batch[sidx:(sidx+1), :] - prediction, 2)]).squeeze().cpu()
    images_in=all_images[:,t_in]
    images_out=all_images[:,t_out]

    ncols  = [nframes_in, nframes_out]
    nrows = 3 # gt, pred, mse
    w, h = figaspect(nrows/(nframes_in+nframes_out))
    fig = plt.figure(figsize=(w,h), constrained_layout=False)
    outer = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.03, hspace=0.0, width_ratios=ncols)

    ylabels=['GT', 'MODEL', 'MSE']

    wspace=0.05
    hspace=0.05
    in_frames = gridspec.GridSpecFromSubplotSpec(nrows=nrows, ncols=ncols[0],
                        subplot_spec=outer[0], wspace=wspace, hspace=hspace)
    for i in range(nrows):
        for j in range(ncols[0]):
            ax = plt.Subplot(fig, in_frames[i,j])
            ax.set(xticks=[], yticks=[])
            fig.add_subplot(ax)
            ax.imshow(images_in[i,j],cmap='gray',vmin=0, vmax=1)       
            if j == 0:
                ax.set_ylabel(ylabels[i])
            if i==2:
                if j==0:
                    ax.set_xlabel(f't={t_in[j]+1}')
                else:
                    ax.set_xlabel(t_in[j]+1)
            if i==0 and j==len(t_in)//2:
                ax.set_title('Prediction with Inputs')

    out_frames = gridspec.GridSpecFromSubplotSpec(nrows=nrows, ncols=ncols[1],
                        subplot_spec=outer[1], wspace=wspace, hspace=hspace)
    for i in range(nrows):
        for j in range(ncols[1]):
            ax = plt.Subplot(fig, out_frames[i,j])
            ax.set(xticks=[], yticks=[])
            fig.add_subplot(ax)
            ax.imshow(images_out[i,j],cmap='gray',vmin=0, vmax=1)
            if i==2:
                ax.set_xlabel(t_out[j]+1)
            if i==0 and j==len(t_out)//2:
                ax.set_title('Video Prediction without Further Inputs')
    return fig

def log_video(pl_module, batch):
    start = time.time()
    fig = create_video(pl_module, batch)
    pl_module.logger.experiment.log({'video_rollout': wandb.Image(fig)})
    plt.close()
    print(f'Saving Video. Creation time: {time.time()-start:.2f}')

class NewBestModelCallback(pl.callbacks.base.Callback):
    # logs the best loss and makes 
    def __init__(self):
        super().__init__()
        self.best_loss = np.Inf

    def on_train_start(self, trainer, pl_module):
        print('Grabbing batches for video creation')
        self.batches = {}
        self.batches['train'] = next(iter(pl_module.train_dataloader()))
        test_dloaders = pl_module.test_dataloader()
        self.batches['test'] = next(iter(test_dloaders[0]))

    def on_train_epoch_end(self, trainer, pl_module):
        losses = trainer.logger_connector.callback_metrics
        monitor = pl_module.hparams.monitor+'_epoch'
        if (losses[monitor] < self.best_loss):
            self.best_loss = losses[monitor]
            for k, v in losses.items():
                if not 'grad' in k:
                    trainer.logger.experiment.summary[f'best/{k}'] = v
        if ((pl_module.current_epoch+1) % 10)==0:
	        log_video(pl_module, self.batches['test'])


class PeriodicCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, every: int):
        super().__init__(period=-1)
        self.every = every

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if (pl_module.current_epoch+1) % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"latest-{pl_module.current_epoch}.ckpt"
            # prev = (
            #     Path(self.dirpath) / f"latest-{pl_module.current_epoch - self.every}.ckpt"
            # )
            trainer.save_checkpoint(current)
            # prev.unlink(missing_ok=True)