import torch
import gpytorch
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import BetaLikelihood
# from gpytorch.priors import SmoothedBoxPrior
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal


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
                 train_x, 
                 train_y, 
                 likelihood,
                 mean_module,
                 covar_module):
        
        super(ExactGPModel, self).__init__(train_x, 
                                           train_y, 
                                           likelihood)
        self.train_x = train_x
        self.train_y = train_y
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

    def _train(self, n_iter, lr, optim):
        """
        Train the GP model

        Args:
            n_iter (int): number of iterations
            lr (float): learning rate
            optim (str): optimizer
        """
        
        self.train()
        self.likelihood.train()
        
        if optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError('optim must be Adam or SGD')
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
       
        for i in range(n_iter):
            
            optimizer.zero_grad()
            
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            
            optimizer.step()
            
            if i % 100 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))

    def predict(self, x_test):
        """ 
        Make prediction on test data

        Args:
            x_test (torch.Tensor): test data
        
        Returns:
            preds_test (torch.Tensor): predicted mean test
        """
        self.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            preds_test = self.likelihood(self(x_test))
            preds_train = self.likelihood(self(self.train_x))
        
        return preds_test, preds_train
    

# TODO make Beta likelihood in a (Sparse) Variational GP model

class ApproximateGPBaseModel(ApproximateGP):
    """ 
    Base model for performing inference with a Gaussian Process using
    stochastic variational inference with inducing points

    Args:
        train_x (torch.Tensor): training data
        likelihood (gpytorch.likelihoods.Likelihood): likelihood
    """

    def __init__(self, train_x : torch.Tensor, likelihood : gpytorch.likelihoods.Likelihood):
        
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution) 
        
        super(ApproximateGPBaseModel, self).__init__(variational_strategy)
        
        dims = 1 if len(train_x.shape) == 1 else train_x.shape[1]
        print(f'Number of dimensions: {dims}')
        print(f'Number of inducing points: {train_x.size(0)}')
        
        self.mean_module = ConstantMean() #prior=SmoothedBoxPrior(0.5, 1))
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dims))

        self.likelihood = likelihood
    
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
        k = self.covar_module(mu)
        latent_u = MultivariateNormal(mu, k)

        return latent_u
        
    def fit(self, n_iter : int, lr : float, optim : torch.optim.Optimizer, device : torch.device, verbose : bool = True):
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
                                            num_data=y.size(0))
        
        print_freq = n_iter // 10
        
        for i in range(n_iter):
            optimizer.zero_grad()
            output = self(self.X)
            loss = -elbo(output, self.y)
            loss.backward()
            optimizer.step()

            if verbose and i % print_freq == 0:
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
            preds = self.likelihood(self(X))
            
        return preds

class BetaGP(ApproximateGPBaseModel):
    """ 
    Stochastic Variational Inference GP model for regression using
    inducing points for scalability and a Beta likelihood for bounded outputs
    in the range [0, 1]

    Args:
        inducing_points (torch.Tensor): inducing points
        variational_dist (gpytorch.variational.VariationalDistribution): variational distribution
        mean_module (gpytorch.means.Mean): mean module
        covar_module (gpytorch.kernels.Kernel): covariance module
    """
    # inherits init from ApproximateGPBaseModel but adds X and y to input when initialising
    def __init__(self, X : torch.Tensor, y : torch.Tensor):
        self.X = X
        self.y = y
        self.model = super(BetaGP, self).__init__(X, BetaLikelihood(scale_constraint=gpytorch.constraints.Interval(-0.01, 1.01)))
