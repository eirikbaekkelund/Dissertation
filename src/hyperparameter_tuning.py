import torch
import gpytorch
import optuna

from gpytorch.kernels import MaternKernel, ScaleKernel
from src.models import BetaGP, ExactGPModel
from src.beta_likelihood import BetaLikelihood_MeanParametrization

class HyperParameterOptimization:
    """ 
    Hyperparameter optimization using Optuna.

    Args:
        model (gpytorch.models.Model): model to optimize
        x_train (torch.Tensor): training inputs
        y_train (torch.Tensor): training targets
        x_test (torch.Tensor): test inputs
        y_test (torch.Tensor): test targets
    """

    def __init__(self, 
                 model : str,
                 x_train : torch.Tensor,
                 y_train : torch.Tensor,
                 x_test : torch.Tensor,
                 y_test : torch.Tensor,
                ):
        assert model in ['beta', 'exact']
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def sample_params_matern(self,
                             trial : optuna.trial.Trial
                             ):
        """ 
        Sample hyperparameters for the model.

        Args:
            trial (optuna.trial.Trial): Optuna
        
        Returns:
            dict: dictionary of hyperparameters
        """
        
        # sample hyperparameters
        
        lengthscale_shape = trial.suggest_float('lengthscale_shape', 1, 10, step=1)
        lengthscale_rate = trial.suggest_float('lengthscale_rate', 1, 10, step=1)

        signal_bound = trial.suggest_float('signal_bound', 0.01, 0.9, step=0.05)
        
        # create kernel
        matern = MaternKernel(nu=3/2,
                              lengthscale_prior=gpytorch.priors.GammaPrior(lengthscale_shape, lengthscale_rate),
                              lengthscale_constraint=gpytorch.constraints.Positive()
                                )
        scaled_matern = ScaleKernel(matern,
                                    outputscale_prior=gpytorch.priors.GammaPrior(5, 2),
                                    outputscale_constraint=gpytorch.constraints.Interval(signal_bound, 1)
                                    )
        return scaled_matern
    
    def sample_params_likelihood(self,
                                 trial : optuna.trial.Trial
                                ):
        """
        Sample hyperparameters for the likelihood.

        Args:
            trial (optuna.trial.Trial): Optuna
        
        Returns:
            likelihood (gpytorch.likelihoods.Likelihood): likelihood of the model
        """
        likelihood_scale = trial.suggest_float('likelihood_scale', 5, 100, step=5)
        likelihood_correcting_scale = trial.suggest_float('likelihood_correcting_scale', 1, 5, step=1)
        
        likelihood = BetaLikelihood_MeanParametrization(scale=likelihood_scale,
                                                        correcting_scale=likelihood_correcting_scale,
                                                        lower_bound=0.03,
                                                        upper_bound=0.97)
                                                        
        return likelihood                                      

    def get_config(self,
                   config : dict,
                   kernel : gpytorch.kernels.Kernel,
                   likelihood : gpytorch.likelihoods.Likelihood,
                   jitter : float,
                   idx : int = None):
        """
        Get the configuration of the model.

        Args:
            config (dict): configuration of the model
            kernel (gpytorch.kernels.Kernel): kernel of the model
            likelihood (gpytorch.likelihoods.Likelihood): likelihood of the model
            jitter (float): jitter for the cholesky decomposition
        
        Returns:
            inputs (dict): dictionary of inputs for the model
        """
        config = {
            'type': 'stochastic',
            'name': 'mean_field',
            'num_inducing_points': self.x_train.size(0),
            # TODO maybe change mean_init_std to a prior
            'mean_init_std': 1,
        }
        
        if idx is not None:
            y = self.y_train[:, idx]
        else:
            y = self.y_train

        inputs = {
            'X': self.x_train,
            'y': y,
            'mean_module': gpytorch.means.ConstantMean(),
            'covar_module': kernel,
            'likelihood': likelihood,
            'config': config,
            'jitter': jitter
        }

        return inputs
    

    def train(self,
              inputs : dict,
              trial : optuna.trial.Trial,
              idx : int = None
              ):
        """ 
        Train the model with the sampled hyperparameters.

        Args:
            inputs (dict): dictionary of inputs for the model
            trial (optuna.trial.Trial): Optuna
        
        Returns:
            float: negative log likelihood
        """
        if self.model == 'beta':
            model = BetaGP(**inputs)
        elif self.model == 'exact':
            model = ExactGPModel(**inputs)
        
        #n_iter = trial.suggest_int('n_iter', 100, 500, step=100)
        lr = trial.suggest_float('lr', 0.01, 0.5)

        model.fit(n_iter=300, 
                  lr=lr, 
                  optim=torch.optim.Adam, 
                  device=torch.device('cpu'),
                  verbose=False)
        
        with torch.no_grad():
            trained_pred_dist = model.likelihood(model(self.x_test))
        
        if idx is not None:
            nlpd = gpytorch.metrics.negative_log_predictive_density(trained_pred_dist, self.y_test[:, idx])
        else:
            nlpd = gpytorch.metrics.negative_log_predictive_density(trained_pred_dist, self.y_test)
        
        # if negative infinity, replace with 0, happens when the 
        # likelihood is close to 0 or 1 where we have scaled the beta mean
        
        nlpd = torch.where(torch.isinf(nlpd), torch.zeros_like(nlpd), nlpd)

        # msll = gpytorch.metrics.mean_standardized_log_loss(self.y_test, trained_pred_dist.mean)
        return nlpd.median()
    
    def get_loss(self,
                trial : optuna.trial.Trial,
                config : dict,
                kernel : gpytorch.kernels.Kernel,
                likelihood : gpytorch.likelihoods.Likelihood,
                jitter : float,
                idx : int = None
                ):
        """
        Loss function that handle the PSD not positive error for 
        ill-conditioned matrices.
        """
        try:
            inputs = self.get_config(config, kernel, likelihood, jitter, idx)
            loss = self.train(inputs, trial, idx)
        except:
            inputs = self.get_config(config, kernel, likelihood, jitter*10, idx)
            loss = self.train(inputs, trial, idx)
        
        return loss
        
    def objective(self,
                  trial : optuna.trial.Trial,
                  config : dict,
                  jitter : float
                 ):
        """ 
        Objective function for Optuna.

        Args:
            trial (optuna.trial.Trial): Optuna
        
        Returns:
            float: negative log predictive density
        """
        kernel = self.sample_params_matern(trial)
        likelihood = self.sample_params_likelihood(trial)
        
        if len(self.y_train.shape) > 1: 
            
            losses = []
            
            for i in range(self.y_train.shape[1]):
                loss = self.get_loss(trial, config, kernel, likelihood, jitter, i)
                losses.append(loss)
            
            return torch.mean(torch.tensor(losses, dtype=torch.float32))
        
        else:
            loss = self.get_loss(trial, config, kernel, likelihood, jitter)
            return loss
