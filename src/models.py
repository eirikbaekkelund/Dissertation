from abc import ABC
import torch
import gpytorch

from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import (VariationalStrategy, 
                                  LMCVariationalStrategy, 
                                  IndependentMultitaskVariationalStrategy)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from src.variational_dist import VariationalBase


########################################
#######  Exact GP Model Class ##########
########################################

class ExactGPModel(ExactGP):
    """ 
    Class for exact GP model.

    Args:
        train_x (torch.Tensor): training data
        train_y (torch.Tensor): training labels
        likelihood (gpytorch.likelihoods.Likelihood): likelihood
        mean_module (gpytorch.means.Mean): mean module
        covar_module (gpytorch.kernels.Kernel): covariance module
    """
    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor,
                 likelihood : gpytorch.likelihoods.Likelihood,
                 mean_module : gpytorch.means.Mean,
                 covar_module : gpytorch.kernels.Kernel):
        
        super(ExactGPModel, self).__init__(X, 
                                           y, 
                                           likelihood)
        self.X = X
        self.y = y
        self.mean_module = mean_module
        self.covar_module = covar_module
    
    def forward(self, x):
        """ 
        Model prediction of the GP model

        Args:
            x (torch.Tensor): input data
        
        Returns:
            gpytorch.distributions.MultivariateNormal: GP model 
                                                       f ~ GP( m(x), k(x, x)) 
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, 
            n_iter : int,
            lr : float,
            optim : torch.optim.Optimizer,
            device : torch.device):
        """
        Train the GP model

        Args:
            n_iter (int): number of iterations
            lr (float): learning rate
            optim (str): optimizer
            device (torch.device): device to train on
        """
        self.to(device)
        
        self.train()
        self.likelihood.train()
        
        optimizer = optim(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        print_freq = n_iter // 10
        self.losses = []


        for i in range(n_iter):
            
            optimizer.zero_grad()
            
            output = self(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())
            
            if (i + 1) % print_freq == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))

    def predict(self, X : torch.Tensor, device : torch.device):
        """ 
        Make predictions with the GP model

        Args:
            X (torch.Tensor): test data
            device (torch.device): device to make predictions on

        Returns:
            preds (gpytorch.distributions.Posterior): predictive posterior
        """
        self.eval()
        self.likelihood.eval()
        
        self.to(device)
        self.likelihood.to(device)

        with torch.no_grad():
            preds = self.likelihood(self(X)) 
            
        return preds
    

########################################
####  Approximate GP Model Classes  ####
########################################

# TODO maybe use UnwhitenedVariationalStrategy instead of VariationalStrategy for having exact inducing points
# TODO possibly add natural gradients
# TODO config to work for unwhitened, natural, and tril natural
# TODO create fit function for natural variational distribution
# TODO add X and y as input params to ApproximateGPBaseModel and remove BetaGP and GaussianGP

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
            assert y.min() >= 0 and y.max() <= 1, 'y must be in the range [0, 1] for Beta likelihood'
        assert X.size(0) == y.size(0), 'X and y must have same number of data points'
        
        # add perturbation to the data to avoid numerical issues for bounded outputs
        if y.min() == 0:
            y += jitter
        
        if y.max() == 1:
            y -= jitter
        
        self.X = X
        self.y = y
        
        
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
            optim : torch.optim.Optimizer,
            device : torch.device, 
            verbose : bool = True):
        """
        Train the GP model using SVI

        Args:
            n_iter (int): number of iterations
            lr (float): learning rate
            optim (str): optimizer
            device (torch.device): device to train on
            verbose (bool): whether to print training progress
        """
        self.to(device)
        
        self.train()
        self.likelihood.train()

        optimizer = optim(self.parameters(), lr=lr)
        elbo = gpytorch.mlls.VariationalELBO(likelihood=self.likelihood, 
                                            model=self, 
                                            num_data=self.y.size(0))
        
        print_freq = n_iter // 10
        self.losses = []
        
        for i in range(n_iter):
            optimizer.zero_grad()
            output = self(self.X)
            loss = -elbo(output, self.y)
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())

            if verbose and (i+1) % print_freq == 0:
                print(f'Iter {i+1}/{n_iter} - Loss: {loss.item():.3f}')
    
    def predict(self, X : torch.Tensor, device : torch.device):
        """ 
        Make predictions with the GP model

        Args:
            X (torch.Tensor): test data
            device (torch.device): device to make predictions on

        Returns:
            preds (gpytorch.distributions.MultivariateNormal): predictive mean and variance
        """
        self.eval()
        self.likelihood.eval()
        
        self.to(device)
        self.likelihood.to(device)

        with torch.no_grad():
           
            if not isinstance(self.likelihood, GaussianLikelihood):
                with gpytorch.settings.num_likelihood_samples(30):
                    # TODO if beta likelihood then predict using the mode
                    # the mode should give the most likely value for the prediction
                    preds = self.likelihood(self(X)) 
            else:
                preds = self.likelihood(self(X))
            
        return preds

# TODO implement MultiTaskModel (variational and exact)
    
########################################
##########  MultiTaskModel  ############
########################################



# TODO implement Latent Force Model

########################################
##########  Latent Force Model  ########
########################################
