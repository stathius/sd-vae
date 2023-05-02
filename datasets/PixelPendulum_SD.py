import torch
from torch.utils.data import Dataset
import deepdish as dd
import pandas as pd
import numpy as np
import h5py

class PixelPendulumDataset_SD(Dataset):
    def __init__(self, filename, num_frames, noise_std=None):
        self.filename = filename
        self.num_frames = num_frames
        self.noise_std=noise_std

        self.h5 = h5py.File(filename, 'r')
        self.trajectories = self.h5.get('frames')
        self.labels = self.h5.get('labels')
        self.max_len = self.trajectories.shape[1]

        self.label_names = ['g', 'initial_angle', 'initial_velocity', 'pendulum_length']

    def get_labels_min_max(self):
        mins = []
        maxs = []
        for k in self.label_names:
            mins.append(self.labels[k][:].min())
            maxs.append(self.labels[k][:].max())

        return  torch.tensor(mins, requires_grad=False), \
                torch.tensor(maxs, requires_grad=False) 

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        labels = []
        for k in self.label_names:
            labels.append(self.labels[k][idx])
        labels_array = np.concatenate(labels)

        max_start = self.max_len - self.num_frames + 1
        start = np.random.choice(range(max_start))
        trajectory = self.trajectories[idx, start:start+self.num_frames]

        if self.noise_std is not None:
            trajectory += np.random.randn(*trajectory.shape) * self.noise_std
            trajectory = np.clip(trajectory, 0.0, 1.0)

        trajectory = np.expand_dims(trajectory, axis=1)
        return { 'trajectory': trajectory,
                 'labels': labels_array
                 }