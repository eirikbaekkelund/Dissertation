import torch
import gpytorch
from gpytorch.distributions import base_distributions

class BetaLikelihood_MeanParametrization(gpytorch.likelihoods.BetaLikelihood):
    
    def __init__(self, scale, correcting_scale, lower_bound, upper_bound, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        assert scale > 0, 'scale must be positive'
        assert correcting_scale > 0, 'scale must be positive'
        assert 0 <= lower_bound <= 1, 'lower bound must be in [0, 1]'
        assert 0 <= upper_bound <= 1, 'upper bound must be in [0, 1]'
        assert lower_bound < upper_bound, 'lower bound must be smaller than upper bound'
        
        self.scale = scale
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.correcting_scale = correcting_scale

    def forward(self, function_samples, *args, **kwargs):
        
        mixture = torch.distributions.Normal(0, 1).cdf(function_samples)
        eps = 1e-9
        alpha = mixture * self.scale + eps
        beta = self.scale - alpha + eps

        # corrects the alpha and beta parameters if the mixture is close to the bounds
        self.alpha = torch.where((mixture > self.lower_bound) | (mixture < self.upper_bound), self.correcting_scale * alpha, alpha)
        self.beta = torch.where((mixture > self.lower_bound) | (mixture < self.upper_bound), self.correcting_scale * beta, beta)

        return base_distributions.Beta(concentration1=self.alpha, concentration0=self.beta)