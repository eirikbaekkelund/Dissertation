import torch
import gpytorch
from gpytorch.kernels import (MaternKernel, 
                              RBFKernel,
                              PeriodicKernel,
                              ScaleKernel, 
                              AdditiveKernel, 
                              ProductKernel)
from gpytorch.means import ZeroMean
from gpytorch.priors import Prior
from gpytorch.constraints import Positive, Interval
from typing import Optional

class Kernel:
    """ 
    Class for creating kernels based on specifications

    Args:
        num_latent (int, optional): number of latent functions. Defaults to 1.
        Particularly used for LMC in multitask GP.
    """
    def __init__(self, num_latent: int = 1):
        self.num_latent = num_latent

    def get_matern(self,
                   lengthscale_constraint : Optional[Positive] = None,
                   outputscale_constraint : Optional[Positive] = None,
                   lengthscale_prior: Optional[Prior] = None,
                   outputscale_prior: Optional[Prior] = None,
                   nu: float = 3/2,):
        """
        Returns a Matern kernel with the specified constraints and priors
        """
        return ScaleKernel( MaternKernel(nu=nu,
                                    lengthscale_constraint=lengthscale_constraint,
                                    lengthscale_prior=lengthscale_prior,
                                    batch_shape=torch.Size([self.num_latent])),
                            outputscale_constraint=outputscale_constraint,
                            outputscale_prior=outputscale_prior)

    def get_periodic(self,
                    lengthscale_constraint : Optional[Positive] = None,
                    outputscale_constraint : Optional[Positive] = None,
                    periodic_constraint : Optional[Positive] = None,
                    lengthscale_prior: Optional[Prior] = None,
                    outputscale_prior: Optional[Prior] = None,
                    period_prior: Optional[Prior] = None):
        """
        Returns a Periodic kernel with the specified constraints and priors
        """
        return ScaleKernel(PeriodicKernel(
                            lengthscale_constraint=lengthscale_constraint,
                            lengthscale_prior=lengthscale_prior,
                            period_constraint=periodic_constraint,
                            period_prior=period_prior,
                            batch_shape=torch.Size([self.num_latent])),
                        outputscale_constraint=outputscale_constraint,
                        outputscale_prior=outputscale_prior)
    
    def get_quasi_periodic(self,
                           base: gpytorch.kernels.Kernel,
                           matern_quasi: gpytorch.kernels.MaternKernel,
                           periodic1: gpytorch.kernels.PeriodicKernel,
                           periodic2: Optional[gpytorch.kernels.PeriodicKernel] = None):
        """
        Returns a combined Periodic kernel with the specified constraints and priors
        """
        
        if periodic2 is not None:
            periodic_kernel = ProductKernel(periodic1, periodic2)

        else:
            periodic_kernel = periodic1
        
        product = ProductKernel(periodic_kernel, matern_quasi)
        quasi_periodic = AdditiveKernel(product, base)
        
        quasi_periodic.batch_shape = torch.Size([self.num_latent])

        return quasi_periodic

def get_mean_covar(num_latent : int = 1, base_kernel='matern'):
    """
    Returns the mean and kernel for the GP model for one dimensional input
    using temporal kernel (quasi-periodic kernel) 

    Args:
        num_latent (int, optional): number of latent functions. Defaults to 1.
        Particularly used for LMC in multitask GP (otherwise it is the same as num_tasks)
    
    Returns:
        mean (ZeroMean): zero mean function
        covar (ScaleKernel): kernel
    """
    mean = ZeroMean(batch_shape=torch.Size([num_latent]))

    kernel = Kernel(num_latent=num_latent)
    if base_kernel == 'matern':
        base = kernel.get_matern(lengthscale_constraint=Positive(initial_value=3),
                                    outputscale_constraint=Positive(initial_value=0.4))
    elif base_kernel == 'rbf':
        base = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latent]),
                                    lengthscale_constraint=Positive(initial_value=3)))
    
    matern_quasi = kernel.get_matern(lengthscale_constraint=Positive(initial_value=100),
                                    outputscale_constraint=Positive(initial_value=0.1))
    periodic1 = kernel.get_periodic(outputscale_constraint=Positive(initial_value=0.1),
                                    periodic_constraint=Positive(initial_value=1.5),
                                    lengthscale_constraint=Positive(initial_value=2))

    covar = kernel.get_quasi_periodic(base=base, 
                                        matern_quasi=matern_quasi,
                                        periodic1=periodic1)
    
    return mean, covar

def get_mean_covar_weather(
        num_latents : int, 
        d : int,
        weather_kernel : str = 'rbf',
        combine : str = 'product',
        use_ard_dim : bool = False):
    """ 
    Get the mean and kernel for Hadamard GP

    Args:
        num_latents (int): number of latent functions if LMC otherwise number of tasks
        d : number of input dimensions
    
    Returns:
        mean (ZeroMean): zero mean function
        covar (ScaleKernel): Hadamard kernel
    """
    assert combine in ['product', 'sum'], "combine must be either 'product' or 'sum'"
    assert weather_kernel in ['rbf', 'matern'], "weather_kernel must be either 'rbf' or 'matern'"
    mean, covar_t = get_mean_covar(num_latent=num_latents)
    if weather_kernel == 'rbf':
        if use_ard_dim:
            covar_w = ScaleKernel(RBFKernel(batch_shape=torch.Size([num_latents]),
                                ard_num_dims=d-1,
                                has_lengthscale=True,
                                lengthscale_constraint=Positive(initial_value=0.1)))
        else:
            covar_w = ScaleKernel(RBFKernel(batch_shape=torch.Size([num_latents]),
                                lengthscale_constraint=Positive(initial_value=0.1)))
    else:
        if use_ard_dim:
            covar_w = ScaleKernel(MaternKernel(nu=3/2, batch_shape=torch.Size([num_latents]),
                            ard_num_dims=d-1,
                            has_lengthscale=True,
                            lengthscale_constraint=Positive(initial_value=0.2)))
        else:
            covar_w = ScaleKernel(MaternKernel(nu=3/2, batch_shape=torch.Size([num_latents]),
                            lengthscale_constraint=Positive(initial_value=0.2)))
    
    # set active dimension based on exogenous part and temporal part
    covar_w.active_dims = torch.tensor([i for i in range(d-1)])
    covar_t.active_dims = torch.tensor([d-1])

    if combine == 'sum':
        return mean, AdditiveKernel(covar_w, covar_t)

    return mean, ProductKernel(covar_w, covar_t)