from abc import ABC
import numpy as np
from torch.utils.data import Dataset

class LFMDataset(Dataset):

    def __getitem__(self, index):
        return self.data[index]

    @property
    def num_outputs(self):
        """The number of LFM outputs."""
        return self._num_outputs

    @num_outputs.setter
    def num_outputs(self, value):
        self._num_outputs = value

    @property
    def data(self):
        """
        List of data points, each a tuple(a, b).
        For time-series, a and b are 1-D.
        For spatiotemporal series, a is (2, T) corresponding to a row for time and space, and b is 1-D.
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def __len__(self):
        return len(self.data)

class TranscriptomicTimeSeries(LFMDataset, ABC):
    def __init__(self):
        self._m_observed = None
        self._t_observed = None

    @property
    def t_observed(self):
        return self._t_observed

    @t_observed.setter
    def t_observed(self, value):
        self._t_observed = value

    @property
    def m_observed(self):
        """m_observed has shape (replicates, genes, times)"""
        return self._m_observed

    @m_observed.setter
    def m_observed(self, value):
        self._m_observed = value

class PV_LFM_Dataset(TranscriptomicTimeSeries):
    """ 
    Construct a dataset for the LFM model using data 
    from the PVDataLoader class.

    Args:
        num_outputs (int): number of PV systems
        m_observed (torch.Tensor): observed data (pv systems)
        f_observed (torch.Tensor): observed data (latent function)
        train_t (torch.Tensor): observed time
        variance (torch.Tensor): variance of the data
    """
    def __init__(self, num_outputs, m_observed, f_observed, train_t, variance, **kwargs):
        
        assert m_observed.shape[1] == num_outputs
        assert len(train_t) == m_observed.shape[0] == f_observed.shape[0]
       
        super().__init__()

        self.num_outputs = m_observed.shape[1] if len(m_observed.shape) > 1 else 1
        self.f_observed = f_observed.view(1, 1, len(train_t))
        self.t_observed = train_t
        self.variance = variance
        self.names = np.array(['PV System ' + str(i) for i in range(num_outputs)])

        self.data = [(train_t, m_observed[:, i]) for i in range(num_outputs)]

