import gpytorch
import torch
import numpy as np
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.models import ApproximateGP
from variational_dist import VariationalBase
from gpytorch.variational import (VariationalStrategy, 
                                  LMCVariationalStrategy, 
                                  IndependentMultitaskVariationalStrategy)
from beta_likelihood import BetaLikelihood_MeanParametrization
from data_loader import (PVDataLoader, 
                             periodic_mapping,
                             train_test_split
)

# set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class MultiTaskBetaGP(ApproximateGP):
    """ 
    Base model for performing inference with a Multitask Gaussian Process (GP) using
    stochastic variational inference (SVI). The model has several outputs and a 
    Beta likelihood is used for each output to account for bounded data.
    """
    def __init__(self,
                 X : torch.Tensor,
                 y : torch.Tensor,
                 likelihood : gpytorch.likelihoods.Likelihood,
                 mean_module : gpytorch.means.Mean,
                 covar_module : gpytorch.kernels.Kernel,
                 variational_strategy : str,
                 num_latents : int,
                 config : dict,
                 jitter : float = 1e-4,
                 learn_inducing_locations : bool = False
                 ):
        if isinstance(likelihood, gpytorch.likelihoods.BetaLikelihood):
            assert y.min() >= 0 and y.max() <= 1, 'y must be in the range [0, 1] for Beta likelihood'
        
        assert X.size(1) == y.size(0), 'X and y must have same number of data points'
        assert variational_strategy in ['lmc', 'mt_indep'], 'Variational strategy must be either lmc or mt_indep'

        # add perturbation to the data to avoid numerical issues for bounded outputs
        if y.min() == 0:
            y += jitter
        
        if y.max() == 1:
            y -= jitter
        
        self.X = X
        self.y = y
        
        variational_dist = VariationalBase(config).variational_distribution
        base_variational = VariationalStrategy( self,
                                                inducing_points=X,
                                                variational_distribution=variational_dist,
                                                learn_inducing_locations=learn_inducing_locations,
                                                jitter_val=jitter)

        if variational_strategy == 'lmc':
            variational_strategy = LMCVariationalStrategy(base_variational_strategy=base_variational,
                                                          num_tasks=y.size(1),
                                                          num_latents=num_latents,
                                                          jitter_val=jitter)
        
        elif variational_strategy == 'mt_indep':
            variational_strategy = IndependentMultitaskVariationalStrategy(base_variational_strategy=base_variational,
                                                                           num_tasks=y.size(1),
                                                                           jitter_val=jitter)

        super(ApproximateGP, self).__init__(variational_strategy)

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood
    
    def forward(self, x, **kwargs):
        """ 
        Forward pass through the model
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return MultitaskMultivariateNormal(mean_x, covar_x)

    def predict(self, X, device : torch.device):
        """ 
        Make predictions with the model
        """
        self.to(device)
        
        self.eval()
        self.likelihood.eval()

        if not isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood):
                with gpytorch.settings.num_likelihood_samples(30):
                    # TODO if beta likelihood then predict using the mode
                    # the mode should give the most likely value for the prediction
                    preds = self.likelihood(self(X)) 
        else:
            preds = self.likelihood(self(X))
        
        return preds

    def get_inducing_points(self):
        """ 
        Get inducing points
        """
        return self.variational_strategy.inducing_points
    
    def fit(self,
            n_iter : int,
            lr : float,
            optim : torch.optim.Optimizer,
            device : torch.device,
            verbose : bool = True
            ):
        """
        Fit the model using SVI
        """
        self.to(device)
        
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
                        {'params': self.parameters()},
                        {'params': self.likelihood.parameters()},
                        ], lr=lr)
        
        elbo = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=self.y.size(0))

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
                print(f'Iter({i+1}/{n_iter}) - Loss: {loss.item():.3f}')


if __name__ == '__main__':
    # data parameters
    DAY_INIT = 10
    DAY_MIN = 8
    DAY_MAX = 16
    N_DAYS = 5
    MINUTE_INTERVAL = 5
    DAILY_DATA_POINTS = (DAY_MAX - DAY_MIN) * 60 / MINUTE_INTERVAL
    N_HOURS_PRED = 2
    N_SYSTEMS = 15
    RADIUS = 0.35
    COORDS = (55, -1.5)
    IDX = 6

    loader = PVDataLoader(n_days=N_DAYS,
                    day_init=DAY_INIT,
                    n_systems=N_SYSTEMS,
                    radius=RADIUS,
                    coords=COORDS,
                    minute_interval=MINUTE_INTERVAL,
                    day_min=DAY_MIN,
                    day_max=DAY_MAX,
                    folder_name='pv_data',
                    file_name_pv='pv_data_clean.csv',
                    file_name_location='location_data_clean.csv')
    
    time, y = loader.get_time_series()
    if y.max() > 1:
        y[y > 1] = 1
    
    periodic_time = periodic_mapping(time, DAY_MIN, DAY_MAX, MINUTE_INTERVAL)
    x = torch.stack([periodic_time, time], dim=1)

    # standardize time dimension 
    x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()

    x_train, y_train, x_test, y_test = train_test_split(x, y, N_HOURS_PRED)

    x_train = x_train.repeat(y.size(1), 1, 1).reshape(y.size(1), -1, 2)
    x_test = x_test.repeat(y.size(1), 1, 1).reshape(y.size(1), -1, 2)

    matern_base = MaternKernel(nu=3/2, 
                      ard_num_dims=x.shape[1], 
                      lengthscale_prior=gpytorch.priors.GammaPrior(2, 8),
                      lengthscale_constraint=gpytorch.constraints.Positive()
                      )

    scaled_matern = ScaleKernel(matern_base, 
                                outputscale_prior=gpytorch.priors.GammaPrior(5, 2),
                                outputscale_constraint=gpytorch.constraints.Interval(0.1, 1)
                                )
    jitter = 1e-4
    config = {  'type': 'stochastic',
                'name': 'mean_field',
                'num_inducing_points': x_train,
                'batch_shape': torch.Size([y_train.size(1)]),
                'mean_init_std': 1,
                }

    multitask_model = MultiTaskBetaGP(  X=x_train,
                                        y=y_train,
                                        likelihood=BetaLikelihood_MeanParametrization(scale=15,
                                                                                    correcting_scale=1,
                                                                                    lower_bound=0.10,
                                                                                    upper_bound=0.80),
                                        mean_module=gpytorch.means.ConstantMean(),
                                        covar_module=scaled_matern,
                                        variational_strategy='lmc',
                                        num_latents=1,
                                        config=config,
                                        jitter=jitter,
                                        learn_inducing_locations=False
                                    )