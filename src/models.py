import torch
import gpytorch
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, UnwhitenedVariationalStrategy
from gpytorch.likelihoods import BetaLikelihood, GaussianLikelihood
from gpytorch.priors import SmoothedBoxPrior, UniformPrior, NormalPrior, GammaPrior, HalfCauchyPrior
from gpytorch.kernels import MaternKernel, PeriodicKernel, RBFKernel, ScaleKernel, AdditiveKernel, ProductKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal


########################################
#######  Exact GP Model Classes ########
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

    def fit(self, n_iter, lr, optim):
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
    

########################################
####  Approximate GP Model Classes  ####
########################################

# TODO maybe use UnwhitenedVariationalStrategy instead of VariationalStrategy for having exact inducing points
# TODO make Beta likelihood work with SVI
# TODO add option for covariance function
# TODO make work for additive and product kernels
# TODO possibly add natural gradients

class ApproximateGPBaseModel(ApproximateGP):
    """ 
    Base model for performing inference with a Gaussian Process using
    stochastic variational inference with inducing points

    Args:
        train_x (torch.Tensor): training data
        likelihood (gpytorch.likelihoods.Likelihood): likelihood
        variational_distribution (gpytorch.variational.VariationalDistribution): variational distribution
    """

    def __init__(self, train_x : torch.Tensor, 
                 likelihood : gpytorch.likelihoods.Likelihood, 
                 variational_distribution : gpytorch.variational):
       
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution, learn_inducing_locations=True) 
        
        super(ApproximateGPBaseModel, self).__init__(variational_strategy)
        
        dims = 1 if len(train_x.shape) == 1 else train_x.shape[1]
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=3/2, ard_num_dims=dims))

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
        k = self.covar_module(x)

        return MultivariateNormal(mu, k)
        
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
                                            num_data=self.y.size(0))
        
        print_freq = n_iter // 10
        
        for i in range(n_iter):
            optimizer.zero_grad()
            output = self(self.X)
            loss = -elbo(output, self.y)
            loss.backward()
            optimizer.step()

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
            preds = self.likelihood(self(X))
            
        return preds

class GaussianGP(ApproximateGPBaseModel):
    """ 
    Stochastic Variational Inference GP model for regression using
    inducing points for scalability and a Gaussian likelihood for unbounded outputs

    Args:
        inducing_points (torch.Tensor): inducing points
        variational_dist (gpytorch.variational.VariationalDistribution): variational distribution
        mean_module (gpytorch.means.Mean): mean module
        covar_module (gpytorch.kernels.Kernel): covariance module
    """
    def __init__(self, X : torch.Tensor, y : torch.Tensor):
        super(GaussianGP, self).__init__(X, GaussianLikelihood(), CholeskyVariationalDistribution(X.size(0)) )
        self.X = X
        self.y = y

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
    def __init__(self, X : torch.Tensor, y : torch.Tensor):
        super(BetaGP, self).__init__(X, BetaLikelihood(), CholeskyVariationalDistribution(X.size(0)) )
        self.X = X
        self.y = y
