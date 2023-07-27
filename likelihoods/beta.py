import torch
import gpytorch
import numpy as np
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.constraints import Positive, Interval
from gpytorch.priors import Prior
from typing import Optional

class BetaLikelihood_MeanParametrization(gpytorch.likelihoods.BetaLikelihood):
    
    def __init__(self, scale, correcting_scale=1, lower_bound=0, upper_bound=1, *args, **kwargs):
        
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
        
        # TODO possibly adjust scale for multitask Beta likelihood
        alpha = mixture * self.scale + eps
        beta = self.scale - alpha + eps

        # corrects the alpha and beta parameters if the mixture is close to the bounds
        self.alpha = torch.where((mixture > self.lower_bound) | (mixture < self.upper_bound), self.correcting_scale * alpha, alpha)
        self.beta = torch.where((mixture > self.lower_bound) | (mixture < self.upper_bound), self.correcting_scale * beta, beta)

        return base_distributions.Beta(concentration1=self.alpha, concentration0=self.beta)
    
    def mode(self):
        """ 
        Calculate the mode of a beta distribution given the alpha and beta parameters

        Args:
            alpha (torch.Tensor): alpha parameter
            beta (torch.Tensor): beta parameter
        
        Returns:
            result (torch.Tensor): modes of the beta distribution drawn from MC samples
        """
        # detach alpha and beta from the graph
        result = np.zeros_like(self.alpha)  # Initialize an array of zeros with the same shape as alpha

        mask_alpha_gt_1 = self.alpha > 1
        mask_beta_gt_1 = self.beta > 1
        mask_alpha_eq_beta = self.alpha == self.beta
        mask_alpha_le_1 = self.alpha <= 1
        mask_beta_le_1 = self.beta <= 1

        result[mask_alpha_gt_1 & mask_beta_gt_1] = (self.alpha[mask_alpha_gt_1 & mask_beta_gt_1] - 1) / (self.alpha[mask_alpha_gt_1 & mask_beta_gt_1] + self.beta[mask_alpha_gt_1 & mask_beta_gt_1] - 2)
        result[mask_alpha_eq_beta] = 0.5
        result[mask_alpha_le_1 & mask_beta_gt_1] = 0
        result[mask_alpha_gt_1 & mask_beta_le_1] = 1

        return result

class MultitaskBetaLikelihood(BetaLikelihood_MeanParametrization):
    """ 
    A multitask BetaLikelihood that supports multitask GP regression.
    """
    def __init__(
        self,
        n_tasks: int,
        scale = 15,
        batch_shape: torch.Size = torch.Size([]),
        scale_prior: Optional[Prior] = None,
        scale_constraint: Optional[Interval] = None,
    ) -> None:
        super().__init__(scale)

        if scale_constraint is None:
            scale_constraint = Positive()

        self.raw_scale = torch.nn.Parameter(torch.ones(*batch_shape, 1, n_tasks))
        if scale_prior is not None:
            self.register_prior("scale_prior", scale_prior, lambda m: m.scale, lambda m, v: m._set_scale(v))

        self.register_constraint("raw_scale", scale_constraint)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        ret = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        
        num_event_dim = len(function_dist.event_shape)
        
        if num_event_dim > 1:  # Do appropriate summation for multitask likelihood
            ret = ret.sum(list(range(-1, -num_event_dim, -1)))
        return ret