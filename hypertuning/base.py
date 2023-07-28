from abc import ABC, abstractmethod
import torch
import gpytorch
import optuna
from kernels.quasi_periodic import generate_quasi_periodic
from gpytorch.kernels import (MaternKernel,
                              ScaleKernel,
                              PeriodicKernel,
                              AdditiveKernel)
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.constraints import Positive, Interval

class HyperOptBase(ABC):
    """ 
    Base class for hyperparameter optimization.
    """

    def __init__(self, 
                 train_loader : torch.utils.data.DataLoader,
                 test_loader : torch.utils.data.DataLoader):
        assert len(train_loader) == len(test_loader), 'train and test loader must have the same length'

        self.train_loader = train_loader
        self.test_loader = test_loader
    
    @abstractmethod
    def sample_params(self, trial : optuna.trial.Trial):
        pass

    @abstractmethod
    def instantiate_model(self, y_tr, y_te, trial : optuna.trial.Trial):
        pass

    @abstractmethod
    def objective(self, trial : optuna.trial.Trial):
        pass

class GPQuasiPeriodic(HyperOptBase):
    """ 
    Hyperparameter optimization using Optuna for a Quasi-Periodic GP model.

    Args:
        train_loader (torch.utils.data.DataLoader): training data loader
        test_loader (torch.utils.data.DataLoader): test data loader
        num_latents (int, optional): number of latent functions. Defaults to 1.
    """

    def __init__(self, 
                train_loader : torch.utils.data.DataLoader,
                test_loader : torch.utils.data.DataLoader,
                num_latents : int = 1,
                ):
        
        assert len(train_loader) == len(test_loader), 'train and test loader must have the same length'
        super().__init__(train_loader=train_loader, test_loader=test_loader)
        self.num_latents = num_latents

    def sample_params_quasi_periodic(self, trial : optuna.trial.Trial):
        # TODO: add gamma priors
        pass
    
    def sample_mean(self, 
                    trial : optuna.trial.Trial):
        """
        Sample the mean of the model.
        """
        mean_type = trial.suggest_categorical('mean_type', ['constant', 'zero'])

        if mean_type == 'constant':
            return gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_latents]))
        else: 
            return gpytorch.means.ZeroMean(batch_shape=torch.Size([self.num_latents]))

    def sample_params(self, trial : optuna.trial.Trial):
        """ 
        Sample the model parameters.
        """
        kernel = self.sample_params_quasi_periodic(trial)
        mean = self.sample_mean(trial)

        return mean, kernel
    
    @abstractmethod
    def get_likelihood(self, trial : optuna.trial.Trial):
        pass
            
    @abstractmethod
    def instantiate_model(self, x_tr, y_tr, trial : optuna.trial.Trial):
        pass
    
    @abstractmethod
    def metric(self, y_dist, target):
        pass

    @abstractmethod    
    def objective(self, trial : optuna.trial.Trial):
        pass