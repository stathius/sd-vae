import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size, output_size, model_size, latent_size, nonlinearity, coord_dim, use_layer_norm):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.coord_dim = coord_dim
        self.model_size = model_size
        self.latent_size = latent_size
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(input_size * coord_dim, model_size[0])
        self.fc2 = nn.Linear(model_size[0], model_size[1])
        self.fc3 = nn.Linear(model_size[1], latent_size)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(latent_size)
        self.fc4 = nn.Linear(latent_size, model_size[1])
        self.fc5 = nn.Linear(model_size[1], model_size[0])
        self.fc6 = nn.Linear(model_size[0], output_size * coord_dim)

        if nonlinearity == 'relu':
            self.activation = nn.ReLU()
        elif nonlinearity == 'tanh':
            self.activation = nn.Tanh()
        elif nonlinearity == 'leaky':
            self.activation = nn.LeakyReLU()
        else:
            raise('Unknown nonlinearity. Accepting relu or tanh only.')

    def forward(self, x):
        batch_size = list(x.size())[0]
        x = x.reshape(x.shape[0], -1).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        latents = self.activation(self.fc3(x))
        if self.use_layer_norm:
            latents = self.layer_norm(latents)
        x = self.activation(self.fc4(latents))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)

        output_size = [batch_size, self.output_size, self.coord_dim]
        return x.reshape(output_size), latents