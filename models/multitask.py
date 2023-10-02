import torch
import gpytorch
import numpy as np
import wandb
from gpytorch.variational import (VariationalStrategy, 
                                  LMCVariationalStrategy,
                                  MeanFieldVariationalDistribution,
                                  CholeskyVariationalDistribution)
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from data.utils import store_gp_module_parameters
from likelihoods import MultitaskBetaLikelihood

class MultitaskGPModel(ApproximateGP):
    def __init__(self,
                 X : torch.Tensor,
                 y : torch.Tensor,
                 likelihood : gpytorch.likelihoods.Likelihood = None,
                 mean_module : gpytorch.means.Mean = None,
                 covar_module : gpytorch.kernels.Kernel = None,
                 num_latents : int = 4,
                 learn_inducing_locations : bool = False,
                 variational_dist : str = 'cholesky',
                 jitter : float = 1e-6):
        # check that num_latents is consistent with the batch_shape of the mean and covar modules
        assert num_latents == mean_module.batch_shape[0], 'num_latents must be equal to the batch_shape of the mean module'
        assert num_latents == covar_module.batch_shape[0], 'num_latents must be equal to the batch_shape of the covar module'
        assert variational_dist in ['mean_field', 'cholesky'], 'variational_dist must be one of: mean_field, cholesky'

        num_tasks = y.size(-1)
        
        if y.max() >= 1:
            y[y >= 1] = 1 - 1e-6
        if y.min() <= 0:
            y[y <= 0] = 1e-6

        if variational_dist == 'mean_field':
            variational_distribution = MeanFieldVariationalDistribution(
                                        num_inducing_points=X.size(0)[::2], 
                                        batch_shape=torch.Size([num_latents]),
                                        jitter=jitter
                                    )
        elif variational_dist == 'cholesky':
            # MeanField constructs a variational distribution for each output dimension
            variational_distribution = CholeskyVariationalDistribution(
                                        num_inducing_points=X.size(0), 
                                        batch_shape=torch.Size([num_latents]),
                                        jitter=jitter
                                    )
        
        # LMC constructs MultitaskMultivariateNormal from the base var dist
        variational_strategy = LMCVariationalStrategy(
                            VariationalStrategy(
                                    model=self, 
                                    inducing_points=X, 
                                    variational_distribution=variational_distribution, 
                                    learn_inducing_locations=learn_inducing_locations,
                                    jitter_val=jitter
                                ),
                            num_tasks=num_tasks,
                            num_latents=num_latents,
                            latent_dim=-1,
                            jitter_val=jitter
                        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = mean_module
        self.covar_module =  covar_module
        self.likelihood = likelihood
        self.X = X
        self.y = y
        self.jitter = jitter
        
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

        self.losses = []

        if use_wandb:
            wandb.init(
                project ='multitask-gp',
                config={'learning_rate': lr, 'epochs': n_iter}
            )
        
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=self.y.size(0))
        
        params = [p[1] for p in self.named_parameters() if 'likelihood' not in p[0]]
        likelihood_params = [p[1] for p in self.named_parameters() if 'likelihood' in p[0]]
        optim = [torch.optim.Adam(params, lr=lr), torch.optim.Adam(likelihood_params, lr=lr)]
        
        print_freq = n_iter // 10
        self.bad_run = False

        for i in range(n_iter):
            
            [opt.zero_grad() for opt in optim]
            output = self(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            [opt.step() for opt in optim]

            self.losses.append(loss.item())
            
            if verbose and (i+1) % print_freq == 0:
                print(f'Iter {i+1}/{n_iter} - Loss: {loss.item()}')
            
            # too high beta dispersion break and decrease dispersion
            if self.losses[-1] > 20 and i > 150:
                self.bad_run = True
                print(f'Loss too high at iter {i+1} - Decreasing beta dispersion')
                break
            
            if use_wandb:
                log_dict = store_gp_module_parameters(self)
                log_dict['loss'] = loss.item()
                wandb.log(log_dict)
            
            # if loss is not decreasing for 15 iterations, stop training
            if i > 0:
                if abs(self.losses[-2] - self.losses[-1]) < 1e-2:
                    j += 1
                    if j == 15:
                        print(f'Early stopping at iter {i+1}')
                        break
                else:
                    j = 0
        
        if use_wandb:
            wandb.finish()
        
        
    
    def get_inducing_points(self):
        return self.variational_strategy.base_variational_strategy.inducing_points
    
    def predict_mean(self, dist):
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

    def predict(self, x, pred_type='dist'):
        """ 
        Get the predictions for the given x values.
        The prediction type can be one of: dist, median, mean, mode, all or
        one can get the posterior predictive distribution.

        Args:
            x (torch.Tensor): input tensor
            pred_type (str, optional): prediction type. Defaults to 'median'.
        
        Returns:
            dist (torch.distributions.Distribution) if pred_type is 'dist': the posterior predictive distribution
            (pred, lower, upper) (torch.Tensor, torch.Tensor, torch.Tensor) if pred_type is 'median', 'mean', 'mode'
            
            where pred is the prediction of the given type and lower, upper 
            is the 95% confidence interval from MC sampling from the predictive distribution.
    
        """
        assert pred_type in ['dist', 'median', 'mean', 'mode', 'all'], 'pred_type must be one of: dist, median, mean, mode, all'
        self.eval()
        self.likelihood.eval()
        if isinstance(self.likelihood, gpytorch.likelihoods.MultitaskGaussianLikelihood):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                dist = self.likelihood(self(x))
                mean = dist.mean
                lower, upper = dist.confidence_region()

                return mean, lower, upper
        
        elif isinstance(self.likelihood, gpytorch.likelihoods.BetaLikelihood):
           with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.num_likelihood_samples(30):
                
                dist = self.likelihood(self(x))
                if pred_type == 'dist':
                    return dist

                samples = dist.sample(sample_shape=torch.Size([100]))
                lower, upper = self.confidence_region(samples)

                if pred_type == 'median':
                    median = self.predict_median(samples)
                    return median, lower, upper
                
                elif pred_type == 'mean':
                    mean = self.predict_mean(dist)
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
    
    def warm_start(self, model_dict):
        """ 
        Warm start the model with the given model_dict.
        """
        try:
            self.load_state_dict(model_dict)
            try:
                # load all parameters except base_variational_strategy variational parameters as they are not compatible
                # due to mismatch in number of training points              
                for name, param in model_dict.items():
                    if 'base_variational_strategy' not in name:
                        self.state_dict[name].copy_(param)
            except Exception as e:
                print(e)
                
                print('Could not load all parameters')
        except Exception as e:
            print(e)
            print('Could not load state dict')
