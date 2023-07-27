from torch.utils.data import Dataset

class PVDataLoader(Dataset):
    """ 
    Custom dataset for the PV data to use from the CV folds

    Args:
        X (torch.tensor): list of input data
        y (torch.tensor): list of target data
    """
    def __init__(self, X : list, y : list):
        assert len(X) == len(y), 'X and y must have the same length'
        self.x = X
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)