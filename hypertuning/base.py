from abc import ABC, abstractmethod
import torch
import optuna
from kernels.quasi_periodic import generate_quasi_periodic
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.metrics import negative_log_predictive_density as nlpd
from likelihoods.beta import BetaLikelihood_MeanParametrization, MultitaskBetaLikelihood

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
    
    def run_study(self, n_trials : int = 100, direction : str = 'minimize'):
        """ 
        Create the Optuna study.
        """
        self.study = optuna.create_study(direction=direction)
        self.study.optimize(self.objective, n_trials=n_trials)
        return self.study
    
    @abstractmethod
    def sample_params(self, trial : optuna.trial.Trial):
        """
        Sample the model parameters.
        """

    @abstractmethod
    def instantiate_model(self, y_tr, y_te, trial : optuna.trial.Trial):
        """ 
        Create the model.
        """

    @abstractmethod
    def objective(self, trial : optuna.trial.Trial):
        """ 
        Objective function for the hyperparameter optimization.
        """
       
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

    def get_quasi_periodic(self, trial : optuna.trial.Trial):
        """
        Sample the parameters of the quasi-periodic kernel.
        """
        alpha_matern = trial.suggest_int('alpha_matern', 1, 10, step=1)
        beta_matern = trial.suggest_int('beta_matern', 5, 15, step=1)
        
        alpha_periodic_L = trial.suggest_int('alpha_periodic', 1, 10, step=1)
        beta_periodic_L = trial.suggest_int('beta_periodic', 1, 10, step=1)

        alpha_periodic_P = trial.suggest_int('alpha_periodic_P', 1, 10, step=1)
        beta_periodic_P = trial.suggest_int('beta_periodic_P', 1, 10, step=1)

        covar_module = generate_quasi_periodic(num_latent=self.num_latents,
                                                matern_alpha=alpha_matern,
                                                matern_beta=beta_matern,
                                                periodic_alpha_L=alpha_periodic_L,
                                                periodic_beta_L=beta_periodic_L,
                                                periodic_alpha_P=alpha_periodic_P,
                                                periodic_beta_P=beta_periodic_P)

        return covar_module
    
    def get_mean(self, 
                    trial : optuna.trial.Trial):
        """
        Sample the mean of the model.
        """
        mean_type = trial.suggest_categorical('mean_type', ['constant', 'zero'])

        if mean_type == 'constant':
            return ConstantMean(batch_shape=torch.Size([self.num_latents]))
        else: 
            return ZeroMean(batch_shape=torch.Size([self.num_latents]))
    
    def get_likelihood(self, trial : optuna.trial.Trial):
        scale = trial.suggest_int('scale', 1, 20, step=5)
        correcting_scale = trial.suggest_int('correcting_scale', 1, 3, step=1)

        if self.num_latents == 1:
            return BetaLikelihood_MeanParametrization(scale=scale,
                                                    correcting_scale=correcting_scale)
        else:
            return MultitaskBetaLikelihood(scale=scale,
                                           correcting_scale=correcting_scale,
                                           num_tasks=self.num_tasks)

    def sample_params(self, trial : optuna.trial.Trial):
        """ 
        Sample the model parameters.
        """
        covar = self.get_quasi_periodic(trial)
        mean = self.get_mean(trial)
        likelihood = self.get_likelihood(trial)
        self.inputs['mean_module'] = mean
        self.inputs['covar_module'] = covar
        self.inputs['likelihood'] = likelihood

    def metric(self, y_dist, target):
        """ 
        Compute the metric of interest.
        """
        return nlpd(y_dist, target).median().item()

    @abstractmethod    
    def objective(self, trial : optuna.trial.Trial):
        pass  