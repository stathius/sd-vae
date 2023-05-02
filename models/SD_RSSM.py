import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class Encoder(nn.Module):
    """
    Encoder to embed image observation (3, 64, 64) to vector (1024,)
    """
    def __init__(self, input_channels):
        super().__init__()        
        self.cv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

    def forward(self, obs):
        hidden = F.relu(self.cv1(obs))
        hidden = F.relu(self.cv2(hidden))
        hidden = F.relu(self.cv3(hidden))
        embedded_obs = F.relu(self.cv4(hidden)).reshape(hidden.size(0), -1)
        return embedded_obs


class RecurrentStateSpaceModel(nn.Module):
    """
    This class includes multiple components
    Deterministic state model: h_t+1 = f(h_t, s_t)
    Stochastic state model (prior): p(s_t+1 | h_t+1)
    State posterior: q(s_t | h_t, o_t)
    NOTE: actually, this class takes embedded observation by Encoder class
    min_stddev is added to stddev same as original implementation
    Activation function for this class is F.relu same as original implementation
    """
    def __init__(self, ssm_state_dim, rnn_hidden_dim, pre_distr_dim=None, min_stddev=0.1, act=F.relu):
        super().__init__()
        if pre_distr_dim==None:
            pre_distr_dim = rnn_hidden_dim
        self._min_stddev = min_stddev
        self.act = act
        
         # converts ssm state to rnn input
        self.fc_ssm_2_rnn_in = nn.Linear(ssm_state_dim, pre_distr_dim)
        # then input dim of the rnn is equal to the pre-distribution dim
        self.rnn = nn.GRUCell(pre_distr_dim, rnn_hidden_dim)
        # converts the rnn hidden to the pre-distribution
        self.fc_rnn_hidden_2_pre_distr = nn.Linear(rnn_hidden_dim, pre_distr_dim)
        # convers pre-distribution to ssm state size of the prior (mean and std)
        self.fc_ssm_state_prior_mean = nn.Linear(pre_distr_dim, ssm_state_dim)
        self.fc_ssm_state_prior_stdd = nn.Linear(pre_distr_dim, ssm_state_dim)

        # for the posterior it combines the rnn hidden and the embedded observation to pre-distribution
        # and outputs the distribution size 
        self.fc_rnn_hidden_emb_obs_2_pre_distr = nn.Linear(rnn_hidden_dim + 1024, pre_distr_dim)
        # convers pre-distribution size to ssm state size of the prior (mean and std)
        self.fc_ssm_state_posterior_mean = nn.Linear(pre_distr_dim, ssm_state_dim)
        self.fc_ssm_state_posterior_stdd = nn.Linear(pre_distr_dim, ssm_state_dim)

    def forward(self, s_t, h_t, o_tp1):
        """
        h_t+1 = f(h_t, s_t)
        Return prior p(s_t+1 | h_t+1), posterior p(s_t+1 | h_t+1, o_t+1) and h_t+1
        for model training
        """
        next_state_prior, h_tp1 = self.prior(s_t, h_t)
        next_state_posterior, post_mean, post_std  = self.posterior(h_tp1, o_tp1)
        return next_state_prior, next_state_posterior, h_tp1, post_mean, post_std 

    def prior(self, s_t, h_t):
        """
        h_t+1 = f(h_t, s_t)
        Compute prior p(s_t+1 | h_t+1)
        """
        s_t = self.act(self.fc_ssm_2_rnn_in(s_t))
        h_tp1 = self.rnn(s_t, h_t)
        pre_distr = self.act(self.fc_rnn_hidden_2_pre_distr(h_tp1))
        mean = self.fc_ssm_state_prior_mean(pre_distr)
        stddev = F.softplus(self.fc_ssm_state_prior_stdd(pre_distr)) + self._min_stddev
        return Normal(mean, stddev), h_tp1

    def posterior(self, h_t, o_t):
        """
        Compute posterior q(s_t | h_t, o_t)
        """
        pre_distr = self.act(self.fc_rnn_hidden_emb_obs_2_pre_distr(torch.cat([h_t, o_t], dim=1)))
        mean = self.fc_ssm_state_posterior_mean(pre_distr)
        stddev = F.softplus(self.fc_ssm_state_posterior_stdd(pre_distr)) + self._min_stddev
        return Normal(mean, stddev), mean, stddev


class ObservationModel(nn.Module):
    """
    p(o_t | s_t, h_t)
    Observation model to reconstruct image observation (3, 64, 64)
    from state and rnn hidden state
    """
    def __init__(self, ssm_state_dim, rnn_hidden_dim, output_channels):
        super().__init__()        
        self.fc = nn.Linear(ssm_state_dim + rnn_hidden_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, output_channels, kernel_size=6, stride=2)

    def forward(self, state, rnn_hidden):
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1))
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        hidden = F.relu(self.dc2(hidden))
        hidden = F.relu(self.dc3(hidden))
        obs = self.dc4(hidden)
        return obs