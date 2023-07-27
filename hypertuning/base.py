from abc import ABC, abstractmethod
import torch
import gpytorch
import optuna
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

    def sample_params_matern(self, trial : optuna.trial.Trial):
        """ 
        Sample hyperparameters for the model using a Matern kernel.

        Args:
            trial (optuna.trial.Trial): Optuna
        
        Returns:
            kernel (gpytorch.kernels.Kernel): kernel of the model
        """
        
        # sample hyperparameters
        
        lengthscale_shape = trial.suggest_float('matern_L_shape', 1, 10, step=1)
        lengthscale_rate = trial.suggest_float('matern_L_rate', 1, 10, step=1)

        signal_shape = trial.suggest_float('signal_matern_shape', 1, 10, step=1)
        signal_rate = trial.suggest_float('signal_matern_rate', 1, 10, step=1)

        # create kernel
        matern = MaternKernel(nu=3/2,
                              lengthscale_prior=gpytorch.priors.GammaPrior(lengthscale_shape, lengthscale_rate),
                              lengthscale_constraint=gpytorch.constraints.Positive(),
                              batch_shape=torch.Size([self.num_latents])
                            )
        scaled_matern = ScaleKernel(matern,
                                    outputscale_prior=gpytorch.priors.GammaPrior(signal_shape, signal_rate),
                                    outputscale_constraint=gpytorch.constraints.Interval(0.05, 1),
                                    batch_shape=torch.Size([self.num_latents])
                                )
        return scaled_matern
    
    def sample_params_periodic(self, trial : optuna.trial.Trial):
        """
        Sample hyperparameters for the model using a periodic kernel.
        """
        lengthscale_shape = trial.suggest_float('periodic_L_shape', 1, 10, step=1)
        lengthscale_rate = trial.suggest_float('period_L_rate', 1, 10, step=1)

        period_shape = trial.suggest_float('period_P_shape', 1, 10, step=1)
        period_rate = trial.suggest_float('period__P_rate', 1, 10, step=1)

        signal_periodic_shape = trial.suggest_float('signal_periodic_shape', 1, 10, step=1)
        signal_periodic_rate = trial.suggest_float('signal_periodic_rate', 1, 10, step=1)

        periodic = PeriodicKernel(  lengthscale_prior=gpytorch.priors.GammaPrior(lengthscale_shape, lengthscale_rate),
                                    lengthscale_constraint=gpytorch.constraints.Positive(),
                                    period_prior=gpytorch.priors.GammaPrior(period_shape, period_rate),
                                    period_constraint=gpytorch.constraints.Positive(),
                                    batch_shape=torch.Size([self.num_latents])
                                    )
        scaled_periodic = ScaleKernel(periodic,
                                    outputscale_prior=gpytorch.priors.GammaPrior(signal_periodic_shape, signal_periodic_rate),
                                    outputscale_constraint=gpytorch.constraints.Positive(),
                                    batch_shape=torch.Size([self.num_latents])
                                    )
        return scaled_periodic
    
    
    def sample_params_quasi_periodic(self,trial : optuna.trial.Trial):
        """
        Sample hyperparameters for the model using a quasi-periodic kernel.
        """
        matern = self.sample_params_matern(trial)
        periodic = self.sample_params_periodic(trial)
        prod_kernel = ScaleKernel(matern * periodic,
                                outputscale_prior=gpytorch.priors.GammaPrior(5, 2),
                                outputscale_constraint=gpytorch.constraints.Interval(0.1, 1),
                                batch_shape=torch.Size([self.num_latents])
                                    )
        quasi_periodic = AdditiveKernel(prod_kernel, matern)

        return quasi_periodic
    
    def sample_mean(self, 
                    trial : optuna.trial.Trial):
        """
        Sample the mean of the model.
        """
        mean_type = trial.suggest_categorical('mean_type', ['constant', 'zero'])

        if mean_type == 'constant':
            return gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_latents]))
        elif mean_type == 'zero':
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