import torch
import gpytorch
from models import VariationalBase
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal

class ApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, dataset, mean, covar, likelihood, config):
        
        config['num_inducing_points'] = dataset[0][0].shape[0]
        variational_dist = VariationalBase(config).variational_distribution
        
        variational_strategy = VariationalStrategy(
            self,
            variational_distribution=variational_dist,
            inducing_points=dataset[0][0],
            learn_inducing_locations=False,
        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = mean
        self.covar_module = covar
        self.likelihood = likelihood
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return MultivariateNormal(mean_x, covar_x)
    
    def predict(self, x):
        with torch.no_grad():
            dist = self.likelihood(self(x))
        return dist