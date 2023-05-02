import torch
from torch.utils.data import Dataset
import deepdish as dd
import pandas as pd
import numpy as np

class NBodyDataset(Dataset):
    """
    Creates a data-loader for the wave prop data
    """
    def __init__(self, filename, noise_std=None, indexes=None):
        self.filename = filename
        dataset = dd.io.load(filename)

        if indexes is None:
            indexes = list(range(0,len(dataset['labels'])))
        
        self.trajectories = torch.FloatTensor(dataset['phase_space'])[indexes]
        self.labels = [dataset['labels'][i] for i in indexes]
        
        self.noise_std=noise_std
        if self.noise_std is not None:
            self.trajectories += torch.randn(self.trajectories.size()) * noise_std
    
    def get_labels_min_max(self):
        df = pd.DataFrame(self.labels)
        params = np.vstack(df['params'].values)
        return torch.tensor(params.min(axis=0)[1:], requires_grad=False),  \
                torch.tensor(params.max(axis=0)[1:], requires_grad=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch= { 
                'trajectory': self.trajectories[idx],
                # Remove K1 from learned params
                 'labels': np.array(self.labels[idx]['params'][1:]) 
                 }
        return batch