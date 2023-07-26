from abc import ABC, abstractmethod
import torch
import gpytorch
import optuna

from gpytorch.kernels import (MaternKernel, 
                              PeriodicKernel, 
                              ScaleKernel, 
                              AdditiveKernel)
from models import ApproximateGPBaseModel, MultitaskGPModel
from baselines import (ExponentialSmoothingModel,
                       SimpleExpSmoothingModel,
                       AutoRegressionModel,
                       ARIMAModel,
                       SARIMAXModel,
                       MarkovAutoregressionModel)

from beta_likelihood import BetaLikelihood_MeanParametrization

# set seed for reproducibility
torch.manual_seed(42)

class HyperOptBase(ABC):
    """ 
    Base class for hyperparameter optimization.
    """

    def __init__(self, 
                 train_loader : torch.utils.data.DataLoader,
                 test_loader : torch.utils.data.DataLoader,
                 ):
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

class GPHyperOptBase(HyperOptBase):
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

class HyperOptBetaGP(GPHyperOptBase):
    def __init__(self,
                train_loader : torch.utils.data.DataLoader,
                test_loader : torch.utils.data.DataLoader,
                ):
        super().__init__(train_loader=train_loader,
                         test_loader=test_loader,
                         num_latents=1)
    
    def get_likelihood(self, trial : optuna.trial.Trial):
        """ 
        Sample the likelihood of the model.
        """
        likelihood_scale = trial.suggest_int('likelihood_scale', 1, 60, step=5)        
        likelihood = BetaLikelihood_MeanParametrization(scale=likelihood_scale)
                                                        
        return likelihood         

    def instantiate_model(self, 
                          x_tr : torch.Tensor, 
                          y_tr : torch.Tensor, 
                          jitter : float, 
                          trial : optuna.trial.Trial):
        """ 
        Create a model instance.
        """
        mean_module, kernel = self.sample_params(trial)
        likelihood = self.get_likelihood(trial)
        config = {
            'type' : 'stochastic',
            'name' : 'mean_field',
            'num_inducing_points' : x_tr.size(0),
            'mean_init_std' : 1,
        }
        inputs = {'x_train': x_tr,
                    'y_train': y_tr,
                    'likelihood': likelihood,
                    'mean_module': mean_module,
                    'covar_module': kernel,
                    'config' : config,
                    'jitter': jitter
                    }
        model = ApproximateGPBaseModel(**inputs)
        model.fit(n_iter=10, lr=0.2, verbose=True)
        return model

    def metric(self, y_dist, target):
        """
        Metric to optimize.
        """
        return gpytorch.metrics.negative_log_predictive_density(y_dist, target).median()
    
    def objective(self, trial : optuna.trial.Trial):
        """
        Objective function to minimize.
        """
        losses = []

        # iterate over folds 
        for (x_tr, y_tr), (x_te, y_te) in zip(self.train_loader, self.test_loader):
            # iterate over each series in the fold
            for i in range(y_tr.shape[1]):
                jitter = 1e-4
                # fit model for each series
                try:
                    model = self.instantiate_model(x_tr, y_tr[:,i], jitter, trial)
                
                except:
                    print('Not PSD error, adding jitter')
                    jitter *= 10
                    try:
                        model = self.instantiate_model(x_tr, y_tr[:,i], jitter, trial)
                    except:
                        continue

                # get predictive distribution
                with torch.no_grad():
                    trained_pred_dist = model.likelihood(model(x_te))
                
                # calculate metric
                nlpd = self.metric(trained_pred_dist, y_te[:,i])
                losses.append(nlpd)
        
        return torch.mean(torch.tensor(losses), dtype=torch.float32)


class HyperOptMultitaskGP(GPHyperOptBase):
    def __init__(self,
            train_loader : torch.utils.data.DataLoader,
            test_loader : torch.utils.data.DataLoader,
            num_latents : int = 5
            ):
        super().__init__(train_loader=train_loader,
                         test_loader=test_loader,
                         num_latents=num_latents)
        
    # TODO make beta likelihood for SVI multitask

    def get_likelihood(self, trial : optuna.trial.Trial):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_tasks)
        return likelihood
                                        

    def instantiate_model(self, 
                          x_tr : torch.Tensor, 
                          y_tr : torch.Tensor, 
                          jitter : float, 
                          trial : optuna.trial.Trial):
        """ 
        Create a model instance.
        """
        mean_module, kernel = self.sample_params(trial)
        likelihood = self.get_likelihood(trial)

        inputs = {'x_train': x_tr,
                    'y_train': y_tr,
                    'likelihood': likelihood,
                    'mean_module': mean_module,
                    'covar_module': kernel,
                    'num_latents' : self.num_latents,
                    'jitter': jitter
                    }
        

        model = MultitaskGPModel(**inputs)
        model.fit(n_iter=10, lr=0.2, verbose=True)
        return model

    def metric(self, y_dist, target):
        """
        Metric to optimize.
        """
        return gpytorch.metrics.negative_log_predictive_density(y_dist, target).median()
    
    def objective(self, trial : optuna.trial.Trial):
        """
        Objective function to minimize.
        """
        losses = []

        # iterate over folds 
        for (x_tr, y_tr), (x_te, y_te) in zip(self.train_loader, self.test_loader):
            # iterate over each series in the fold
            self.n_tasks = y_tr.shape[1]
            for i in range(y_tr.shape[1]):
                jitter = 1e-4
                # fit model for each series
                try:
                    model = self.instantiate_model(x_tr, y_tr, jitter, trial)
                
                except:
                    print('Not PSD error, adding jitter')
                    jitter *= 10
                    model = self.instantiate_model(x_tr, y_tr, jitter, trial)

                # get predictive distribution
                with torch.no_grad():
                    trained_pred_dist = model.likelihood(model(x_te))
                
                # calculate metric
                nlpd = self.metric(trained_pred_dist, y_te)
                losses.append(nlpd)
        
        return torch.mean(torch.tensor(losses), dtype=torch.float32)
    
class ExpSmoothingHyperOpt(HyperOptBase):
    """ 
    Hyperparameter optimization using Optuna for an exponential smoothing model.
    """
    def sample_params(self, trial: optuna.trial.Trial):
        """
        Sample hyperparameters for the model.
        """
        smoothing_level = trial.suggest_float('smoothing_level', 0.0, 1.0)
        smoothing_slope = trial.suggest_float('smoothing_slope', 0.0, 1.0)
        damping_slope = trial.suggest_float('damping_slope', 0.0, 1.0)
        return smoothing_level, smoothing_slope, damping_slope
    