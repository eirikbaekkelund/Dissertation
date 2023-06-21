import torch
import gpytorch
from gpytorch.distributions import base_distributions

class BetaLikelihood_MeanParametrization(gpytorch.likelihoods.BetaLikelihood):
    def __init__(self, scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert scale > 0, 'scale must be positive'
        self.scale = scale

    def forward(self, function_samples, *args, **kwargs):
        mixture = torch.distributions.Normal(0, 1).cdf(function_samples)
        
        alpha = mixture * self.scale
        beta = self.scale - alpha

        eps = 1e-9

        self.alpha = alpha + eps
        self.beta = beta + eps

        return base_distributions.Beta(concentration1=self.alpha, concentration0=self.beta)