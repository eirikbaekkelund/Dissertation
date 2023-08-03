import torch
import gpytorch
import numpy as np
import wandb
from gpytorch.variational import (VariationalStrategy, 
                                  LMCVariationalStrategy,
                                  MeanFieldVariationalDistribution)
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from data.utils import store_gp_module_parameters

class MultitaskGPModel(ApproximateGP):
    def __init__(self,
                 x_train : torch.Tensor,
                 y_train : torch.Tensor,
                 likelihood : gpytorch.likelihoods.Likelihood = None,
                 mean_module : gpytorch.means.Mean = None,
                 covar_module : gpytorch.kernels.Kernel = None,
                 num_latents : int = 1,
                 jitter : float = 1e-4):
        # check that num_latents is consistent with the batch_shape of the mean and covar modules
        assert num_latents == mean_module.batch_shape[0], 'num_latents must be equal to the batch_shape of the mean module'
        assert num_latents == covar_module.batch_shape[0], 'num_latents must be equal to the batch_shape of the covar module'

        num_tasks = y_train.size(-1)
        
        # MeanField constructs a variational distribution for each output dimension
        variational_distribution = MeanFieldVariationalDistribution(
            x_train.size(0), batch_shape=torch.Size([num_latents]),jitter=jitter
        )
        
        # LMC constructs MultitaskMultivariateNormal from the base var dist
        variational_strategy = LMCVariationalStrategy(
                            VariationalStrategy(
                                    model=self, 
                                    inducing_points=x_train, 
                                    variational_distribution=variational_distribution, 
                                    learn_inducing_locations=False, 
                                    jitter_val=jitter
                                ),
                            num_tasks=num_tasks,
                            num_latents=num_latents,
                            latent_dim=-1
                        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = mean_module
        self.covar_module =  covar_module
        self.likelihood = likelihood
        self.x_train = x_train
        self.y_train = y_train
        
    def forward(self, x):
    
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
      
        return MultivariateNormal(mean_x, covar_x)
    
    def fit(self, 
            n_iter : int,
            lr : float, 
            verbose : bool = False,
            use_wandb : bool = False):
            
        self.train()
        self.likelihood.train()

        if use_wandb:
            wandb.init(
                project ='dissertation',
                config={'learning_rate': lr, 'epochs': n_iter}
            )
        
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=self.y_train.size(0))
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        
        print_freq = n_iter // 10
        
        for i in range(n_iter):
            
            optim.zero_grad()
            output = self(self.x_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            optim.step()
            
            if verbose and (i+1) % print_freq == 0:
                print(f'Iter {i+1}/{n_iter} - Loss: {loss.item()}')
            
            if use_wandb:
                log_dict = store_gp_module_parameters(self)
                log_dict['loss'] = loss.item()
                wandb.log(log_dict)
        
        if use_wandb:
            wandb.finish()
    
    def predict_mean(self):
        return dist.mean.mean(axis=0)
    
    def predict_mode(self):
        return self.likelihood.mode().mean(axis=0)
    
    def predict_median(self, samples):
        return samples.median(axis=0).values.mean(axis=0)
    
    def confidence_region(self, samples):
        # per MC sample
        lower, upper = np.percentile(samples, [2.5, 97.5], axis=0)
        # across tasks
        lower, upper = lower.mean(axis=0), upper.mean(axis=0)
        return lower, upper

        
    def predict(self, x, pred_type='median'):
        """ 
        Get the predictions for the given x values.
        The prediction type can be one of: dist, median, mean, mode, all or
        one can get the posterior predictive distribution.

        Args:
            likelihood (gpytorch.likelihoods.Likelihood): Likelihood used for the predictions.
        """
        assert pred_type in ['dist', 'median', 'mean', 'mode', 'all'], 'pred_type must be one of: dist, median, mean, mode, all'
    
        if isinstance(self.likelihood, gpytorch.likelihoods.MultitaskGaussianLikelihood):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                dist = self.likelihood(self(x))
                mean = dist.mean
                lower, upper = dist_train.confidence_region()

                return mean_train, lower_train, upper_train
        
        elif isinstance(self.likelihood, gpytorch.likelihoods.BetaLikelihood):
           with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(30):
                
                dist = self.likelihood(self(x))
                if pred_type == 'dist':
                    return dist

                samples = dist.sample(sample_shape=torch.Size([30]))
                lower, upper = self.confidence_region(samples)

                if pred_type == 'median':
                    median = self.predict_median(samples)
                    return median, lower, upper
                
                elif pred_type == 'mean':
                    mean = self.predict_mean()
                    return mean, lower, upper
                
                elif pred_type == 'mode':
                    mode = self.predict_mode()
                    return mode, lower, upper
             
                else:
                    median = self.predict_median(samples)
                    mean = self.predict_mean()
                    mode = self.predict_mode()
                    
                    return median, mean, mode, lower, upper
                    
        else:
            raise NotImplementedError('Likelihood not implemented')

