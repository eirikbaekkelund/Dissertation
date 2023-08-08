import torch
import gpytorch
from gpytorch.kernels import (MaternKernel, 
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

def get_mean_covar(num_latent=1):
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