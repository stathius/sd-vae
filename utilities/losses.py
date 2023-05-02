import torch
from torch.nn import functional as F

def l1_loss(output, target):
    # The reconstruction loss, between the real values and the prediction.
    # Each output dimension is considered independent, conditioned
    # on the latent representation p(x_i|z). Where i traverses the output dimension.
    # Hence the likelihood is  L = Mult_i p(x_i|z)
    # And the log likelihood is log L = Sum_i log p(x_i|z)
    # We then take the average of those log likelihoods for the batch
    # Notice we use L1 for now not L2
    # https://stats.stackexchange.com/questions/323568/help-understanding-reconstruction-loss-in-variational-autoencoder
    # loss = torch.mean(torch.sum(torch.abs(output.flatten(start_dim=1) - 
    #                             target.flatten(start_dim=1)), axis=1),axis=0)
    # mean per output dimension
    # loss = torch.mean(torch.abs(output - target))
    return F.l1_loss(output, target, reduction='mean')

def mse_loss(output, target):
    # mean per output dimension
    return F.mse_loss(output, target, reduction='mean')

def bce_with_logits_loss(output, target):
    # mean per output dimension
    loss_per_pixel = F.binary_cross_entropy_with_logits(output.flatten(1), target.flatten(1), reduction='none')
    loss_per_batch_sample = torch.sum(loss_per_pixel, dim=1)
    loss_per_batch = torch.mean(loss_per_batch_sample)
    return loss_per_batch

def cnn_vae_mse_loss(output, target):
    """Computes the l2 loss. Assuming sigmoid outputs"""
    loss_per_batch_sample = torch.sum(torch.square(target - torch.sigmoid(output)), dim = 1)
    loss_per_batch = torch.mean(loss_per_batch_sample)
    return loss_per_batch
  
def kld_loss(mu, logvar):
    # computes the kld divergence for a VAE model between the normal prior
    # and the posterior q(z|x) of the VAE
    # The KLD is computed for each sample and then averaged over the batch
    loss_per_batch_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
    loss_per_batch = torch.mean(loss_per_batch_sample, dim = 0)
    return loss_per_batch

def geco_constraint(rec_loss, tol):
    # Computes the constraint for the geco algorithm
    return rec_loss - tol

def label_loss(labels_norm, z_mean):
    pass