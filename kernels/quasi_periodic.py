import torch
import gpytorch
from gpytorch.kernels import (MaternKernel, 
                              PeriodicKernel,
                              ScaleKernel, 
                              AdditiveKernel, 
                              ProductKernel)

def generate_quasi_periodic(num_latent : int = 1,
                            matern_scale: int = 1,
                            matern_rate: int = 1,
                            periodic_scale: int = 1,
                            periodic_rate: int = 1,
                            nu: float = 3/2)->gpytorch.kernels.AdditiveKernel:
    """
    Generate a quasi-periodic kernel

    Args:
        num_latent (int): number of latent functions if using multi-task GP
                          (default: 1 for one dimensional GP)
        matern_scale (int): scale of the matern kernel
        matern_rate (int): rate of the matern kernel
        periodic_scale (int): scale of the periodic kernel
        periodic_rate (int): rate of the periodic kernel
        nu (float): smoothness parameter of the matern kernel
    
    Returns:
        quasi-periodic_matern (gpyporch.kernels.AdditiveKernel): quasi-periodic kernel
    """
    matern_base = MaternKernel(nu=3/2, 
                        lengthscale_prior=gpytorch.priors.GammaPrior(2, 8),
                        lengthscale_constraint=gpytorch.constraints.Positive(),
                        batch_shape=torch.Size([num_latent])
                        )
    periodic = PeriodicKernel(period_length_prior=gpytorch.priors.GammaPrior(3, 2),
                              period_length_constraint=gpytorch.constraints.Positive(),
                              batch_shape=torch.Size([num_latent])

                            )
    scaled_periodic = ScaleKernel(periodic,
                                    outputscale_prior=gpytorch.priors.GammaPrior(5, 1),
                                    outputscale_constraint=gpytorch.constraints.Positive(),
                                    batch_shape=torch.Size([num_latent])
                                )
    scaled_matern = ScaleKernel(matern_base, 
                                outputscale_prior=gpytorch.priors.GammaPrior(5, 2),
                                outputscale_constraint=gpytorch.constraints.Interval(0.05, 1),
                                batch_shape=torch.Size([num_latent])
                                )
    product_kernel_matern_periodic = ScaleKernel(periodic * matern_base,
                                outputscale_prior = gpytorch.priors.GammaPrior(5, 2),
                                outputscale_constraint=gpytorch.constraints.Positive(),
                                batch_shape=torch.Size([num_latent])
                                )

    quasi_periodic_matern = AdditiveKernel(product_kernel_matern_periodic, scaled_matern)

    return quasi_periodic_matern