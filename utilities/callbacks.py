import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import figaspect
import torch

def plot_3body(phase_space, ax, ls):
    d = 2
    r1=phase_space[:,:d]
    r2=phase_space[:,d:2*d]
    r3=phase_space[:,2*d:3*d]

    ax.plot(r1[:,0],r1[:,1],color="tab:red", linestyle=ls)
    ax.plot(r2[:,0],r2[:,1],color="tab:green", linestyle=ls)
    ax.plot(r3[:,0],r3[:,1],color="darkblue", linestyle=ls)

    ms = 50
    #Plot the initial positions
    ax.scatter(r1[0,0],r1[0,1],color="tab:red",marker="s",s=ms)
    ax.scatter(r2[0,0],r2[0,1],color="tab:green",marker="s",s=ms)
    ax.scatter(r3[0,0],r3[0,1],color="darkblue",marker="s",s=ms)

    #Plot the final positions
    ax.scatter(r1[-1,0],r1[-1,1],color="tab:red",marker="o",s=ms)
    ax.scatter(r2[-1,0],r2[-1,1],color="tab:green",marker="o",s=ms)
    ax.scatter(r3[-1,0],r3[-1,1],color="darkblue",marker="o",s=ms)

    ax.set_ylim([-5,5])
    ax.set_xlim([-5,5])

def log_prediction_charts(pl_module, batch, stage):
    dataset = pl_module.hparams.dataset

    if ('pixel_pendulum' in dataset):
        rollout_size = np.max(pl_module.val_rec_loss_sizes) 
        output, target = pl_module.rollout(batch, start=0, rollout_size=rollout_size)
        output = torch.sigmoid(output).detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        s = 0
        labels = batch['labels'][s].detach().numpy()

        for frame in pl_module.val_rec_loss_sizes:
            #Create figure
            fig=plt.figure(figsize=(6,4))
            ax=fig.add_subplot(131)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_aspect('equal')
            plt.imshow(target[s][frame-1], cmap='gray')
            plt.title('target')
            
            ax=fig.add_subplot(132)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_aspect('equal')
            plt.imshow(output[s][frame-1], cmap='gray')
            plt.title('output')

            ax=fig.add_subplot(133)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_aspect('equal')
            plt.imshow(np.abs(target[s][frame-1] - output[s][frame-1]), cmap='gray')
            plt.title('abs diff')
            pl_module.log_image(f"visual_{stage}/timestep {frame}", fig)
            plt.close()
    elif ('pendulum' in dataset) or ('lv' in dataset):
        output, target = pl_module.rollout(batch, start=0, rollout_size=500)
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        for s in [0]: # random sample from batch
            w, h = figaspect(1/3.5)
            fig, axs = plt.subplots(1, 3, figsize=(w/2.0,h/2.0))
            labels = batch['labels'][s].detach().numpy()
            labels_str = ' - '.join([f'{l:.2f}' for l in labels])
            plt.suptitle(f"Params: {labels_str}")
            for var in [0, 1]:
                axs[var].plot(target[s][:,var], label=f'target {var}')
                axs[var].plot(output[s][:,var], label=f'prediction {var}')
                axs[var].plot(output[s][:,var]-target[s][:,var], label=f'error {var}')
                axs[var].legend()
            axs[2].plot(target[s][:,0], target[s][:,1], label='target')
            axs[2].plot(output[s][:,0], output[s][:,1], label='prediction')
            axs[2].legend()
            pl_module.log_image(f"visual_{stage}/sample {s}", fig)
            plt.close()       
    elif ('3body' in dataset):
        output, target = pl_module.rollout(batch, start=0, rollout_size=500)
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        for s in [0]:
            labels = batch['labels'][s].detach().numpy()

            #Create figure
            fig=plt.figure(figsize=(5,5))
            ax=fig.add_subplot(111)
            ax.set_aspect('equal')

            #Plot the orbits
            plot_3body(target[s], ax, 'dotted')
            plot_3body(output[s], ax, 'solid')

            # Set the title
            K2, m1, m2, m3 = labels
            title = f'K2 {K2:.2f} - m1/m2/m3 {m1:.2f}/{m2:.2f}/{m3:.2f}'
            
            plt.title(title)
            pl_module.log_image(f"visual_{stage}/sample {s}", fig)
            plt.close()


class BestValidationCallback(pl.callbacks.base.Callback):
    # logs the best validation loss and other stuff
    def __init__(self, monitor, use_wandb):
        super().__init__()
        self.monitor = monitor
        self.best_val_loss = np.Inf
        self.use_wandb = use_wandb


    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or pl_module.hparams.fast_dev_run:
            return
        losses = trainer.logger_connector.callback_metrics
        if (losses[self.monitor] < self.best_val_loss):
            self.best_val_loss = losses[self.monitor]
            if self.use_wandb:
                for k, v in losses.items():
                    if not 'grad' in k:
                        trainer.logger.experiment.summary[f'best/{k}'] = v

            for stage in ['train', 'val']:
                batch = pl_module.batch_sample[stage]
                log_prediction_charts(pl_module, batch, stage)


class TestEndCallback(pl.callbacks.base.Callback):
    # logs the best validation loss and other stuff
    def __init__(self, use_wandb):
        super().__init__()
        self.use_wandb = use_wandb

    def on_test_end(self, trainer, pl_module):
        if self.use_wandb:
            for stage, batch in pl_module.batch_sample.items():
                if 'test' in stage:
                    log_prediction_charts(pl_module, batch, stage)
