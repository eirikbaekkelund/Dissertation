import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """
    Dataset for LSTM model.
    """
    def __init__(self, X, y, sequence_length=30):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if idx >= self.sequence_length - 1:
            idx_start = idx - self.sequence_length + 1
            x = self.X[idx_start:idx+1]
        
        else:
            padding = self.X[0].repeat(self.sequence_length - idx - 1, 1)
            x = self.X[0:idx+1]
            x = torch.cat((padding, x), 0)
        
        return x, self.y[idx]