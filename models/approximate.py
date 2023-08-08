import gpytorch
import torch
import numpy as np
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from models.variational import VariationalBase
from data.utils import store_gp_module_parameters
import wandb

# TODO add parameter tracking

class ApproximateGPBaseModel(ApproximateGP):
    """ 
    Base model for performing inference with a Gaussian Process (GP) using
    stochastic variational inference (SVI) with inducing points which can be
    scaled to large datasets. 
    
    For a guide to GPs and SVI, see the following paper:
    https://arxiv.org/abs/1309.6835

    Args:
        train_x (torch.Tensor): training data
        likelihood (gpytorch.likelihoods.Likelihood): likelihood
        variational_distribution (gpytorch.variational.VariationalDistribution): variational distribution
    """

    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor,
                 likelihood : gpytorch.likelihoods.Likelihood, 
                 mean_module : gpytorch.means.Mean,
                 covar_module : gpytorch.kernels.Kernel,
                 config : dict,
                 jitter : float = 1e-4,
                 learn_inducing_locations : bool = False
                 ):
        
        if isinstance(likelihood, gpytorch.likelihoods.BetaLikelihood):
            if y.max() > 1:
                print('y max, ', y.max())
                y = y / y.max()
            if y.min() < 0:
                y[ y < 0] = 0
                print('y min, ', y.min())

            assert y.min() >= 0 and y.max() <= 1, 'y must be in the range [0, 1] for Beta likelihood'
        assert X.size(0) == y.size(0), 'X and y must have same number of data points'
        
        # avoids numerical issues with Cholesky decomposition
        # can result in all NaNs at first iteration if 0 and 1 are present in y
        if y.min() == 0:
            y[y == 0] += 1e-10

        if y.max() == 1:
            y[y==0] -= 1e-10

        self.X = X
        self.y = y
        self.config = config
        
        
        variational_distribution = VariationalBase(config).variational_distribution
        variational_strategy = VariationalStrategy(self, 
                                                   inducing_points=X, 
                                                   variational_distribution=variational_distribution,
                                                   learn_inducing_locations=learn_inducing_locations,
                                                   jitter_val=jitter) 
        variational_strategy.num_tasks = y.size(1) if len(y.size()) > 1 else 1
        
        super(ApproximateGPBaseModel, self).__init__(variational_strategy)
        
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood
    
    def get_inducing_points(self):
        """ 
        Get inducing points

        Returns:
            torch.Tensor: inducing points
        """
        return self.variational_strategy.inducing_points
    
    
    def warm_start(self, params : dict):
        """ 
        Warm start the model with the given parameters

        Args:
            params (dict): parameters
        """
        self.load_state_dict(params)
    
    def forward(self, x):
        """ 
        Model prediction of the GP model

        Args:
            x (torch.Tensor): input data
        
        Returns:
            latent_u (gpytorch.distributions.MultivariateNormal): GP model 
            f(x), where f ~ GP( m(x), k(x, x))
        """
        mu = self.mean_module(x)
        k = self.covar_module(x)

        return MultivariateNormal(mu, k)
        
    def fit(self, 
            n_iter : int, 
            lr : float,
            verbose : bool = True,
            use_wandb : bool = False,
            objective = 'elbo'):
        """
        Train the GP model using SVI

        Args:
            n_iter (int): number of iterations
            lr (float): learning rate
            optim (str): optimizer
            device (torch.device): device to train on
            verbose (bool): whether to print training progress
        """        
        assert objective in ['elbo', 'pred_log'], 'objective must be one of: elbo, pred_log'
        self.train()
        self.likelihood.train()

        # seperate variational parameters from model parameters during training
        # stochastic gradients for model hyperparameters and likelihood parameters
        # natural gradients for variational parameters

        if self.config['name'] in ['tril_natural', 'natural']:
            variational_ngd_optimizer = gpytorch.optim.NGD(self.variational_parameters(), 
                                                           num_data=self.y.size(0), lr=lr)
            hyperparameter_optimizer = torch.optim.Adam( self.hyperparameters(), lr=0.01)
            natural_grad = True
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            natural_grad = False
        
        if use_wandb:
            wandb.init(
                project ='dissertation',
                config={'learning_rate': lr, 'epochs': n_iter}
            )
        

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if objective == 'pred_log':
            mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood=self.likelihood, 
                                                         model=self, 
                                                         num_data=self.y.size(0))
        else:
            mll = gpytorch.mlls.VariationalELBO(likelihood=self.likelihood, 
                                            model=self, 
                                            num_data=self.y.size(0))
        
        print_freq = n_iter // 10
        losses = []
        # j is tracker for number of iterations without improvement
        j = 0
        for i in range(n_iter):
            if natural_grad:
                variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            
            output = self(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            
            losses.append(loss.item())
            
            if natural_grad:
                variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()
            else:
                optimizer.step()
            
            if use_wandb:
                log_dict = store_gp_module_parameters(self)
                log_dict['loss'] = loss.item()
                wandb.log(log_dict)
            
            if verbose and (i+1) % print_freq == 0:
                print(f'Iter {i+1}/{n_iter} - Loss: {loss.item():.3f}')  
            
            # if loss does not improve by more than 0.1% for 15 iterations, stop training
            if i > 0:
                if abs(losses[-2] - losses[-1]) < 1e-6:
                    j += 1
                    if j == 15:
                        print(f'Early stopping at iter {i+1}')
                        break
                else:
                    j = 0
        
        if use_wandb:
            wandb.finish()
    
    def predict_mean(self, samples):
        """ 
        Mean predictions with the GP model

        Args:
            sanples (torch.Tensor): MC samples from the posterior distribution

        Returns:
            torch.Tensor: posterior mean
        """
        return samples.mean(axis=0).mean(axis=0)
    
    def predict_median(self, samples):
        """ 
        Median predictions with the GP model

        Args:
            samples (torch.Tensor): MC samples from the posterior distribution

        Returns:
            torch.Tensor: posterior median from MC samples
        """
        return samples.median(axis=0).values.mean(axis=0)

    def predict_mode(self):
        """ 
        Mode predictions with the GP model.
        Uses the mode from the posterior:

        \frac{\alpha-1}{\alpha+\beta-2} for α, β > 1
        any value in (0,1) for α, β = 1
        {0, 1} (bimodal) for α, β < 1
        0 for α ≤ 1, β > 1
        1 for α > 1, β ≤ 1
        
        Returns:
            mode (torch.Tensor): posterior mode
        """
        return self.likelihood.mode().mean(axis=0)
    
    def confidence_region(self, samples):
        """ 
        Get the 95% confidence region for the GP model

        Args:
            samples (torch.Tensor): MC samples from the posterior distribution

        Returns:
            torch.Tensor: lower and upper bounds of the 95% confidence region
        """
        lower, upper = np.percentile(samples, [2.5, 97.5], axis=0)
        return lower, upper
    
    def predict(self, x : torch.Tensor, pred_type='dist'):
        """ 
        Make predictions with the GP model

        Args:
            X (torch.Tensor): test data
            pred_type (str, optional): prediction type. Defaults to 'dist'.
        Returns:
            preds (gpytorch.distributions.Distribution): marginalized posterior predictive distribution
            Uses MC sampling for non-Gaussian likelihoods

            Or 
            (pred_type, lower, upper) (torch.Tensor, torch.Tensor, torch.Tensor) 
            if pred_type is 'median', 'mean', 'mode' 

            Or
            (median, mean, mode, lower, upper) (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            if pred_type is 'all'

        """
        assert pred_type in ['dist', 'median', 'mean', 'mode', 'all'], 'pred_type must be one of: dist, median, mean, mode, all'
        self.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            
            # MC samples for non-Gaussian likelihoods
            if not isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood):
                with gpytorch.settings.num_likelihood_samples(30):
                    # get posterior predictive distribution
                    preds = self.likelihood(self(x)) 
                    
                    if pred_type == 'dist':
                        return preds
                    # samples from the posterior predictive distribution
                    samples = preds.sample(sample_shape=torch.Size([30]))
                    lower, upper = self.confidence_region(samples)
                    lower = lower.mean(axis=0)
                    upper = upper.mean(axis=0)

                    if pred_type == 'median':
                        median = self.predict_median(samples)
                        return median, lower, upper
                    
                    elif pred_type == 'mean':
                        mean = self.predict_mean(samples)
                        return mean, lower, upper
                    
                    elif pred_type == 'mode':
                        mode = self.predict_mode()
                        return mode, lower, upper
                    
                    else:
                        median = self.predict_median(samples)
                        mean = self.predict_mean(samples)
                        mode = self.predict_mode()
                        return median, mean, mode, lower, upper

            # Gaussian likelihoods
            else:
                return self.likelihood(self(x))
            