import torch
from torch import nn
import gpytorch
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import VariationalStrategy
from gpytorch.likelihoods import BetaLikelihood, GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from src.variational_dist import VariationalBase


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
# TODO test MeanFieldVariationalDistribution
# TODO test UnwhitenedVariationalStrategy
# TODO test different kernels
# TODO test different likelihoods
# TODO add confidence region for beta likelihood

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

    def __init__(self, train_x : torch.Tensor, 
                 likelihood : gpytorch.likelihoods.Likelihood, 
                 mean_module : gpytorch.means.Mean,
                 covar_module : gpytorch.kernels.Kernel,
                 config : dict,
                 jitter : float = 1e-4,
                 learn_inducing_locations : bool = False
                 ):
        variational_distribution = VariationalBase(config).variational_distribution
        variational_strategy = VariationalStrategy(self, 
                                                   inducing_points=train_x, 
                                                   variational_distribution=variational_distribution,
                                                   learn_inducing_locations=learn_inducing_locations,
                                                   jitter_val=jitter) 
        
        super(ApproximateGPBaseModel, self).__init__(variational_strategy)
        
        # TODO change this to accept exogenous variables
        
        dims = 1 if len(train_x.shape) == 1 else train_x.shape[1]
        
        self.mean_module = mean_module
        self.covar_module = covar_module
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
        
    def fit(self, 
            n_iter : int, 
            lr : float,
            optim : torch.optim.Optimizer,
            device : torch.device, verbose : bool = True):
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
                with gpytorch.settings.num_likelihood_samples(50):
                    preds = self.likelihood(self(X))
            else:
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
    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor, 
                 mean_module : gpytorch.means.Mean,
                 covar_module : gpytorch.kernels.Kernel,
                 config : dict,
                 jitter : float = 1e-6,
                 ):
        
        super(GaussianGP, self).__init__(train_x=X, 
                                         likelihood=GaussianLikelihood(), 
                                         mean_module=mean_module,
                                         covar_module=covar_module,
                                         config=config,
                                         jitter=jitter)
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
    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor,
                 mean_module : gpytorch.means.Mean,
                 covar_module : gpytorch.kernels.Kernel,
                 config : dict,
                 jitter : float = 1e-4,
                 ):
        
        assert y.min() >= 0 and y.max() <= 1, 'y must be in the range [0, 1] for Beta likelihood'
        
        # add perturbation to the data to avoid numerical issues for bounded outputs
        if y.min() == 0:
            y += jitter
        
        if y.max() == 1:
            y -= jitter
        
        super(BetaGP, self).__init__(train_x=X,
                                     likelihood=BetaLikelihood(), 
                                     mean_module=mean_module,
                                     covar_module=covar_module,
                                     config=config,
                                     jitter=jitter)
        self.X = X
        self.y = y

########################################
######  Kalman Filter Smoothing  #######
########################################

# TODO run tests and make sure it works
# TODO add option for exogenous variables
# TODO check if this works for multivariate time series
# TODO set up for device agnostic code

class KalmanFilterSmoother(nn.Module):
    """ 
    Class for performing Kalman filter smoothing on 
    a linear Gaussian state space model.
    
    This is based on the work in: 
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5711863

    Args:
        F (torch.Tensor): transition matrix
        Q (torch.Tensor): process noise covariance matrix
        H (torch.Tensor): observation matrix
        R (torch.Tensor): observation noise covariance matrix
    """
    def __init__(self, F, Q, H, R):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

    def _predict(self, x, P):
        """ 
        Perform the prediction step of the Kalman filter.
        This should be called before the filtering update step.

        Args:
            x: torch.Tensor of shape (d,) where d is the spatial dimension
            P: torch.Tensor of shape (d, d)
        
        Returns:
            xnew: torch.Tensor of shape (d,) where d is the spatial dimension
            Pnew: torch.Tensor of shape (d, d)
        """

        F = self.F
        Q = self.Q
        return torch.matmul(F, x), torch.matmul(torch.matmul(F, P), F.t()) + Q

    def _filter_update(self, x, P, y):
        """ 
        Perform the filtering update step.
        This should be called after applying the forward pass.

        Args:
            x: torch.Tensor of shape (d,) where d is the spatial dimension
            P: torch.Tensor of shape (d, d)
            y: torch.Tensor of shape (d,) where d is the spatial dimension
        
        Returns:
            xnew: torch.Tensor of shape (d,) where d is the spatial dimension
            Pnew: torch.Tensor of shape (d, d)
        """

        R = self.R
        H = self.H
       
        # Compute Kalman gain matrix
       
        if not torch.isnan(y):
            S = torch.matmul(torch.matmul(H, P), H.t()) + R
            chol = torch.cholesky(S)
            
            Kt = torch.cholesky_solve(H @ P, chol)
            Hx = torch.matmul(H, x)
       
            return x + torch.matmul(Kt, y - Hx).t(), P - torch.matmul(torch.matmul(Kt, S.t()), Kt)
       
        else:
            return x, P

    def _smoother_update(self, x_now, x_next, x_forecast, P_now, P_next, P_forecast):
        """ 
        Perform the smoothing update step. 
        This should be called after applying the forward pass.

        Args:
            x_now: torch.Tensor of shape (d,) where d is the spatial dimension
            x_next: torch.Tensor of shape (d,) where d is the spatial dimension
            x_forecast: torch.Tensor of shape (d,) where d is the spatial dimension
            P_now: torch.Tensor of shape (d, d)
            P_next: torch.Tensor of shape (d, d)
            P_forecast: torch.Tensor of shape (d, d)

        Returns:
            xnew: torch.Tensor of shape (d,) where d is the spatial dimension
            Pnew: torch.Tensor of shape (d, d)
        """

        F = self.F
        
        # Compute smoothing gain
        chol = torch.cholesky(P_forecast)
        Jt = torch.cholesky_solve(F @ P_now, chol)
        
        # Update
        xnew = x_now + torch.matmul(Jt, x_next - x_forecast).t()
        Pnew = P_now + torch.matmul(torch.matmul(Jt, P_next - P_forecast), Jt.t())
        
        return xnew, Pnew

    def forward_pass(self, x, P, y_list):
        """ 
        Calling the forward pass gives us the filtering distribution.
         Args:
            x: torch.Tensor of shape (d,) where d is the spatial dimension
            P: torch.Tensor of shape (d, d)
            y_list: torch.Tensor of shape (N, d) containing a list of observations at N timepoints.
                    Note that when there is no observation at time n, then set y_list[n] = torch.nan
        """
        means = []
        covariances = []
        
        for y in y_list:
            
            x, P = self._filter_update(x, P, y)
            
            means.append(x)
            covariances.append(P)
            
            x, P = self._predict(x, P)
       
        return torch.stack(means), torch.stack(covariances)

    def backward_pass(self, x, P):
        """ 
        Calling the backward pass gives us the smoothing distribution. This should be called after applying the forward pass.
        
        Args:
            x: torch.Tensor of shape (N, d) where N is the number of forward time steps and d is the spatial dimension
            P: torch.Tensor of shape (N, d, d)
        
        Returns:
            means: torch.Tensor of shape (N, d) where N is the number of forward time steps and d is the spatial dimension
            covariances: torch.Tensor of shape (N, d, d)
        """        
        N = x.shape[0]
       
        means = [x[-1]]
        covariances = [P[-1]]
        
        for n in range(N - 2, -1, -1):
            # Forecast
            xf, Pf = self._predict(x[n], P[n])
            # Update
            xnew, Pnew = self._smoother_update(x[n], x[n + 1], xf, P[n], P[n + 1], Pf)
            means.append(xnew)
            covariances.append(Pnew)
       
        return torch.flip(torch.stack(means), [0]), torch.flip(torch.stack(covariances), [0])