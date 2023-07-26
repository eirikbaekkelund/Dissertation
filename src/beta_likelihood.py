import torch
import gpytorch
import numpy as np
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.constraints import Positive

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

class MultitaskBetaLikelihood(_OneDimensionalLikelihood):
    """ 
    A multitask BetaLikelihood that supports multitask GP regression.
    """
    def __init__(self, num_tasks, batch_shape=torch.Size([]), scale_prior=None, scale_constraint=None):
        super().__init__()

        if scale_constraint is None:
            scale_constraint = Positive()

        self.raw_scale = torch.nn.Parameter(torch.ones(*batch_shape, 1, num_tasks))
        if scale_prior is not None:
            self.register_prior("scale_prior", scale_prior, lambda m: m.scale, lambda m, v: m._set_scale(v))

        self.register_constraint("raw_scale", scale_constraint)

    @property
    def scale(self):
        return self.raw_scale_constraint.transform(self.raw_scale)

    @scale.setter
    def scale(self, value):
        self._set_scale(value)

    def _set_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_scale)
        self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))

    def forward(self, function_samples, *args, **kwargs):
        
        mixture = torch.distributions.Normal(0, 1).cdf(function_samples)
        eps = 1e-9
        
        alpha = mixture * self.scale + eps
        beta = self.scale - alpha + eps

        # if you want to bias the alpha and beta parameters to peak the distribution (gaussian-like)
        self.alpha = torch.where((mixture > self.lower_bound) | (mixture < self.upper_bound), self.correcting_scale * alpha, alpha)
        self.beta = torch.where((mixture > self.lower_bound) | (mixture < self.upper_bound), self.correcting_scale * beta, beta)

        return base_distributions.Beta(concentration1=self.alpha, concentration0=self.beta)
    

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        ret = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        
        num_event_dim = len(function_dist.event_shape)
        
        if num_event_dim > 1:  # Do appropriate summation for multitask likelihood
            ret = ret.sum(list(range(-1, -num_event_dim, -1)))
        return ret
    
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