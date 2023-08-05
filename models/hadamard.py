import gpytorch
from gpytorch.models import ApproximateGP
from likelihoods import MultitaskBetaLikelihood
from gpytorch.kernels import IndexKernel
from gpytorch.variational import (VariationalStrategy,
                                  LMCVariationalStrategy,
                                  MeanFieldVariationalDistribution)
from models import MultitaskGPModel


class HadamardGP(MultitaskGPModel):
    
    def __init__(self,
                 X : torch.Tensor,
                 y : torch.Tensor,
                 likelihood : gpytorch.likelihoods.Likelihood = None,
                 mean_module : gpytorch.means.Mean = None,
                 covar_module : gpytorch.kernels.Kernel = None,
                 num_latents : int = 1,
                 learn_inducing_locations : bool = False,
                 jitter : float = 1e-4):

        assert num_latents == mean_module.batch_shape[0], 'num_latents must be equal to the batch_shape of the mean module'
        assert num_latents == covar_module.batch_shape[0], 'num_latents must be equal to the batch_shape of the covar module'

        # TODO check if LMC is necessary/works
        super().__init__(X=X,
                        y=y,
                        likelihood=likelihood,
                        mean_module=mean_module,
                        covar_module=covar_module,
                        num_latents=num_latents,
                        learn_inducing_locations=learn_inducing_locations, 
                        jitter=jitter)
        # Since the index kernel does some scaling, the passed 
        # covar module should possibly not scale to avoid overparameterization

        # TODO check if this should have number of latents or number of tasks
        self.task_covar_module = IndexKernel(batch_shape=torch.Size([num_latents]), rank=1)
    
    def forward(self, x, i):
        
        mean_x = self.mean_module(x)
        
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)
        covar = covar_x.mul(covar_i)
        
        return MultivariateNormal(mean_x, covar)
   