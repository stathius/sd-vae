# Adapted from Laura Kulowski
import numpy as np
import random
import sys
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class lstm_encoder(nn.Module):
    def __init__(self, coord_dims, hidden_size, num_layers):
        super().__init__()
        self.coord_dims = coord_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # define LSTM layer
        self.lstm = nn.LSTM(input_size=coord_dims, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)

    def forward(self, x_input):
        '''
        : param x_input:               input of shape (seq_len, # in batch, coord_dims)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence '''
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden

class lstm_decoder(nn.Module):    
    def __init__(self, coord_dims, hidden_size, num_layers):
        super().__init__()
        self.coord_dims = coord_dims
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=coord_dims, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, coord_dims)           

    def forward(self, x_input, encoder_hidden_states):             
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        output = self.linear(lstm_out)    
        return output, self.hidden


class S2S_LSTM(nn.Module):    
    def __init__(self, coord_dims, hidden_size, num_layers, ):
        '''
        : param coord_dims:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''        
        super().__init__()
        self.coord_dims = coord_dims
        self.hidden_size = hidden_size
        self.encoder = lstm_encoder(coord_dims=coord_dims, hidden_size = hidden_size, 
                                    num_layers=num_layers)
        self.decoder = lstm_decoder(coord_dims=coord_dims, hidden_size = hidden_size, 
                                    num_layers=num_layers)

    def forward(self, input, target_len, target=None):        
        '''
        : param input:      input data (seq_len, coord_dims); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''
        outputs = []
        # outputs = torch.zeros(input.shape[0], target_len, input.shape[2])
        encoder_output, encoder_hidden = self.encoder(input) # encode input
        decoder_input = input[:, -1:, :]
        decoder_hidden = encoder_hidden

        for t in range(target_len): 
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # outputs[:,t:(t+1),:] = decoder_output
            outputs += [decoder_output]  # predictions
            if target is None:
                decoder_input = decoder_output
            else:
                decoder_input = target[:,t:(t+1), :]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs