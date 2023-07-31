import torch
import gpytorch
from gpytorch.kernels import (MaternKernel, 
                              PeriodicKernel,
                              ScaleKernel, 
                              AdditiveKernel, 
                              ProductKernel)
from gpytorch.priors import Prior
from typing import Optional

class Kernel:
    """ 
    Class for creating kernels based on specifications
    """
    def __init__(self, num_latent: int = 1):
        self.num_latent = num_latent

    def get_matern(self,
                   lengthscale_constraint : Optional[gpytorch.constraints] = None,
                   outputscale_constraint : Optional[gpytorch.constraints] = None,
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
                    lengthscale_constraint : Optional[gpytorch.constraints] = None,
                    outputscale_constraint : Optional[gpytorch.constraints] = None,
                    periodic_constraint : Optional[gpytorch.constraints] = None,
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
        return AdditiveKernel(product, matern_base)