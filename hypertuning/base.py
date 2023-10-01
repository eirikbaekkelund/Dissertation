from abc import ABC, abstractmethod
import torch
import optuna
from kernels import Kernel
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.metrics import negative_log_predictive_density as nlpd
from gpytorch.constraints import Positive, Interval
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
        torch.save(best_params, f'best_params_{opt_name}.txt')
    
    def sample_train_config(self, trial : optuna.trial.Trial):
        """
        Sample the training configuration.
        """
        lr = 0.1
        epochs = 120

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
      
        self.inputs = {
            'covar_module' : self.get_quasi_periodic(),
            'jitter' : 1e-4,
            'learn_inducing_locations' : False,
        }

    def get_quasi_periodic(self):
        """
        Sample the parameters of the quasi-periodic kernel.
        """
        kernel = Kernel(self.num_latents)
        # TODO  maybe add constraints to the parameters of the kernel
        matern_base = kernel.get_matern(lengthscale_constraint=Positive(),
                                outputscale_constraint=Positive())
        matern_quasi = kernel.get_matern(lengthscale_constraint=Interval(0.3, 1000.0),
                                        outputscale_constraint=Positive())
        
        periodic = kernel.get_periodic(lengthscale_constraint= Positive(),
                                        outputscale_constraint=Positive())
   
        covar_module = kernel.get_quasi_periodic(matern_base=matern_base,
                                                matern_quasi=matern_quasi,
                                                periodic1=periodic)

        return covar_module
    
    def get_mean(self, trial : optuna.trial.Trial):
        """
        Sample the mean of the model.
        """
        mean = trial.suggest_categorical('mean', ['zero', 'constant'])
        if mean == 'zero':
            return ZeroMean(batch_shape=torch.Size([self.num_latents]))
        else:
            return ConstantMean(batch_shape=torch.Size([self.num_latents]))
    
    def get_likelihood(self, trial : optuna.trial.Trial):
        scale_init = trial.suggest_int('scale_init', 1, 50, step=5)
        if self.num_latents == 1:
            return BetaLikelihood_MeanParametrization(scale=scale_init)
        else:
            return MultitaskBetaLikelihood(scale=scale_init,
                                           num_tasks=self.num_tasks)

    def sample_params(self, trial : optuna.trial.Trial):
        """ 
        Sample the model parameters.
        """
        self.inputs['likelihood'] = self.get_likelihood(trial)
        self.inputs['mean_module'] = self.get_mean(trial)
        self.inputs['covar_module'] = self.get_quasi_periodic()
        self.train_config = self.sample_train_config(trial)

    def metric(self, y_dist, target):
        return nlpd(y_dist, target).median().item()

    @abstractmethod    
    def objective(self, trial : optuna.trial.Trial):
        pass  