import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

class VAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int,
                 factors_dim: int, # add extra determistic factors of variatio 
                 hidden_dims: List,
                 coord_dim: int,
                 nonlinearity: str = 'relu',
                 dropout_pct: int = 0.0,
                 use_layer_norm: bool = False,
                 ) -> None:
        super().__init__()

        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == 'leaky':
            self.nonlinearity = nn.LeakyReLU()
        elif nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        else:
            raise('Unknown nonlinearity. Accepting relu or tanh only.')

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.coord_dim = coord_dim # how many coordinatinates each step has
        self.dropout_pct = dropout_pct
        self.use_layer_norm = use_layer_norm
        self.factors_dim = factors_dim

        modules = []

        # Build Encoder
        encoder_dims = [input_dim*self.coord_dim] + hidden_dims
        self.encoder = self.build_modules(encoder_dims, self.nonlinearity, dropout_pct)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dims[-1])
            
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        if self.factors_dim > 0:
            self.fc_factors = nn.Linear(hidden_dims[-1], factors_dim)
        
        hidden_dims.reverse()
        decoder_dims = [latent_dim + factors_dim] + hidden_dims
        self.decoder = self.build_modules(decoder_dims, self.nonlinearity, dropout_pct)


        self.final_layer = nn.Linear(hidden_dims[-1], output_dim*coord_dim)

    @staticmethod
    def build_modules(hidden_dims, nonlinearity, dropout_pct):
        modules = []
        input_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    # nn.BatchNorm1d(h_dim),
                    nonlinearity,
                    # nn.Dropout(dropout_pct)
                    )
            )
            input_dim = h_dim
        return nn.Sequential(*modules)


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x D]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        if self.use_layer_norm:
            result  = self.layer_norm(result)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        # for numerical stability add 1e-5
        # from http://ruishu.io/2018/03/14/vae/
        logvar = self.fc_logvar(result) 

        if self.factors_dim > 0:
            factors = self.fc_factors(result)
            return [mu, logvar, factors]
        else:
            return [mu, logvar, None]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        # result = self.decoder_input(z)
        # result = result.view(-1, 512, 2, 2)
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inpt: Tensor, **kwargs) -> List[Tensor]:
        inpt = inpt.flatten(1).float()
        mu, logvar, factors = self.encode(inpt)
        z = self.reparameterize(mu, logvar)

        if (self.factors_dim > 0):
            assert (factors is not None)
            z = torch.cat((z, factors), axis=1)

        output = self.decode(z)
        output_shape = [inpt.size(0), self.output_dim, self.coord_dim] # B X STEPS X PHASE
        return  [output.reshape(output_shape), mu, logvar, factors]

    def loss_function(self, output: Tensor, target: Tensor, 
                            mu: Tensor, logvar: Tensor,
                      # *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recon_loss = F.l1_loss(output, target, reduction='sum')
        # recon_loss = F.mse_loss(output, target, reduction='sum')

        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss +  kld_loss
        return {'loss': loss, 'recon_loss':recon_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]