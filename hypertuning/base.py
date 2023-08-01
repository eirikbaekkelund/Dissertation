from abc import ABC, abstractmethod
import torch
import optuna
from kernels.kernels import generate_quasi_periodic
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
    
    def save_best_params(self, opt_name : str):
        """ 
        Save the best parameters found by Optuna to a file.

        Args:
            opt_name (str): name of the model
        """
        best_params = self.study.best_params
        best_params['opt_name'] = opt_name
        torch.save(best_params, f'best_params/{opt_name}.txt')
    
    def sample_train_config(self, trial : optuna.trial.Trial):
        """
        Sample the training configuration.
        """
        # sample learning rate between 0.05 and 0.3 with step size 0.05
        lr = trial.suggest_int('lr * 100', 5, 30, step=5) / 100
        # sample number of epochs between 100 and 500 with step size 100
        epochs = trial.suggest_int('epochs', 100, 500, step=100)

        return {'lr' : lr, 'n_iter' : epochs}


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
        
        return covar_module
    
    def get_mean(self, 
                    trial : optuna.trial.Trial):
        """
        Sample the mean of the model.
        """
        mean_type = trial.suggest_categorical('mean', ['constant', 'zero'])

        if mean_type == 'constant':
            return ConstantMean(batch_shape=torch.Size([self.num_latents]))
        else: 
            return ZeroMean(batch_shape=torch.Size([self.num_latents]))
    
    def get_likelihood(self, trial : optuna.trial.Trial):
        bound = trial.suggest_int('scale upper bound', 1, 20, step=5)
        scale = 10 if bound > 10 else bound
        
        if self.num_latents == 1:
            return BetaLikelihood_MeanParametrization(scale=scale,
                                                      scale_upper_bound=bound)
        else:
            return MultitaskBetaLikelihood(scale=scale,
                                           num_tasks=self.num_tasks,
                                           scale_upper_bound=bound)

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
        self.train_config = self.sample_train_config(trial)

    def metric(self, y_dist, target):
        """ 
        Compute the metric of interest.
        """
        return nlpd(y_dist, target).median().item()

    @abstractmethod    
    def objective(self, trial : optuna.trial.Trial):
        pass  