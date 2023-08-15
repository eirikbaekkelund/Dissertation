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
                           matern_base: gpytorch.kernels.MaternKernel,
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
        quasi_periodic = AdditiveKernel(product, matern_base)
        
        quasi_periodic.batch_shape = torch.Size([self.num_latent])

        return quasi_periodic

def get_mean_covar(num_latent : int = 1):
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
    matern_base = kernel.get_matern(lengthscale_constraint=Positive(),
                                    outputscale_constraint=Positive())
    matern_quasi = kernel.get_matern(lengthscale_constraint=Interval(0.3, 1000.0),
                                    outputscale_constraint=Positive())
    periodic1 = kernel.get_periodic(lengthscale_constraint= Positive(),
                                    outputscale_constraint=Positive())

    covar = kernel.get_quasi_periodic(matern_base=matern_base, 
                                        matern_quasi=matern_quasi,
                                        periodic1=periodic1)
    
    return mean, covar

def get_mean_covar_weather(
        num_latents : int, 
        d : int,
        combine : str = 'product'):
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

    mean, covar_t = get_mean_covar(num_latent=num_latents)
    covar_w = ScaleKernel(RBFKernel(batch_shape=torch.Size([num_latents])))
    covar_t = ScaleKernel(covar_t)
    
    # set active dimension based on exogenous part and temporal part
    covar_w.active_dims = torch.tensor([i for i in range(d-1)])
    covar_t.active_dims = torch.tensor([d -1])

    if combine == 'sum':
        return mean, AdditiveKernel(covar_w, covar_t)

    return mean, ProductKernel(covar_w, covar_t)


