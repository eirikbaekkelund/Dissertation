import torch
import gpytorch
import optuna
from torch import nn

from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import VariationalStrategy
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel

from src.variational_dist import VariationalBase
from src.beta_likelihood import BetaLikelihood_MeanParametrization


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
# TODO add option for covariance function
# TODO make work for additive and product kernels
# TODO possibly add natural gradients
# TODO test MeanFieldVariationalDistribution
# TODO test different kernels
# TODO config to work for unwhitened, natural, and tril natural
# TODO create fit function for natural variational distribution

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
                with gpytorch.settings.num_likelihood_samples(100):
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
        X (torch.Tensor): inducing points input data
        y (torch.Tensor): inducing points target data
        variational_dist (gpytorch.variational.VariationalDistribution): variational distribution
        mean_module (gpytorch.means.Mean): mean module
        covar_module (gpytorch.kernels.Kernel): covariance module
        likelihood (gpytorch.likelihoods.Likelihood): likelihood
        config (dict): dictionary of configuration parameters
        jitter (float, optional): jitter value for numerical stability. Defaults to 1e-4.
    """
    def __init__(self, 
                 X : torch.Tensor, 
                 y : torch.Tensor,
                 mean_module : gpytorch.means.Mean,
                 covar_module : gpytorch.kernels.Kernel,
                 likelihood : gpytorch.likelihoods.Likelihood,
                 config : dict,
                 jitter : float = 1e-4,
                 ):
        
        assert y.min() >= 0 and y.max() <= 1, 'y must be in the range [0, 1] for Beta likelihood'
        assert X.size(0) == y.size(0), 'X and y must have same number of data points'
        
        # add perturbation to the data to avoid numerical issues for bounded outputs
        if y.min() == 0:
            y += jitter
        
        if y.max() == 1:
            y -= jitter
        
        self.X = X
        self.y = y
        
        super(BetaGP, self).__init__(train_x=self.X,
                                     likelihood=likelihood, 
                                     mean_module=mean_module,
                                     covar_module=covar_module,
                                     config=config,
                                     jitter=jitter)

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
        matern_nu = trial.suggest_categorical('matern_nu', [3/2, 5/2])
        
        lengthscale_shape = trial.suggest_float('lengthscale_shape', 1, 10, step=1)
        lengthscale_rate = trial.suggest_float('lengthscale_rate', 1, 10, step=1)

        signal_shape = trial.suggest_float('signal_shape', 1, 10, step=1)
        signal_rate = trial.suggest_float('signal_rate', 1, 10, step=1)
        
        # create kernel
        matern = MaternKernel(nu=matern_nu,
                              lengthscale_prior=gpytorch.priors.GammaPrior(lengthscale_shape, lengthscale_rate),
                              lengthscale_constraint=gpytorch.constraints.Positive()
                                )
        scaled_matern = ScaleKernel(matern,
                                    outputscale_prior=gpytorch.priors.GammaPrior(signal_shape, signal_rate),
                                    outputscale_constraint=gpytorch.constraints.Positive()
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
        likelihood_scale = trial.suggest_float('likelihood_scale', 60, 200, step=10)
        likelihood = BetaLikelihood_MeanParametrization(scale=likelihood_scale)

        return likelihood                                      

    def get_config(self,
                   config : dict,
                   kernel : gpytorch.kernels.Kernel,
                   likelihood : gpytorch.likelihoods.Likelihood,
                   jitter : float):
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

        inputs = {
            'X': self.x_train,
            'y': self.y_train,
            'mean_module': gpytorch.means.ConstantMean(),
            'covar_module': kernel,
            'likelihood': likelihood,
            'config': config,
            'jitter': jitter
        }

        return inputs
    

    def train(self,
              inputs : dict,
              trial : optuna.trial.Trial
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
        
        model.predict(self.x_test, device=torch.device('cpu'))

        with torch.no_grad():
            trained_pred_dist = model.likelihood(model(self.x_test))
        
        nlpd = gpytorch.metrics.negative_log_predictive_density(trained_pred_dist, self.y_test)
        # msll = gpytorch.metrics.mean_standardized_log_loss(self.y_test, trained_pred_dist.mean)
        return nlpd.mean()
        
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
            # TODO change to GPyTorch metric
            float: negative log likelihood
        """
        kernel = self.sample_params_matern(trial)
        likelihood = self.sample_params_likelihood(trial)
        inputs = self.get_config(config, kernel, likelihood, jitter)
        
        try:
            inputs = self.get_config(config, kernel, likelihood, jitter)
            loss = self.train(inputs, trial)
        
        # if not PSD --> except the trial, add jitter, and try again
        except:
            inputs = self.get_config(config, kernel, likelihood, jitter*10)
            loss = self.train(inputs, trial)

        return loss