import torch
import gpytorch
import numpy as np
from gpytorch.distributions import base_distributions
from gpytorch.constraints import Positive, Interval
from gpytorch.priors import Prior
from typing import Optional

class BetaLikelihood_MeanParametrization(gpytorch.likelihoods.BetaLikelihood):
    
    def __init__(self, 
                 scale : Optional[torch.Tensor] = 30,
                 scale_upper_bnd : Optional[torch.Tensor] = None,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        assert scale > 0, 'scale must be positive'
        if scale_upper_bnd is not None:
            assert scale_upper_bnd > 0, 'scale_upper_bnd must be positive'
            assert scale_upper_bnd > scale, 'scale_upper_bnd must be greater than scale'
        
        self.scale = scale 
        self.scale_upper_bnd = scale_upper_bnd

    def forward(self, function_samples, *args, **kwargs):
        
        mixture = torch.distributions.Normal(0, 1).cdf(function_samples)
        # set scale by upper bound if passed
        if self.scale_upper_bnd is not None:
            self.scale = torch.clamp(self.scale, 1e-10, self.scale_upper_bnd)

        self.alpha = mixture * self.scale 
        self.beta = self.scale - self.alpha 

        self.alpha = torch.clamp(self.alpha, 1e-10, 1e10)
        self.beta = torch.clamp(self.beta, 1e-10, 1e10)

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
        num_tasks: int,
        scale = 15,
        batch_shape: torch.Size = torch.Size([]),
        scale_prior: Optional[Prior] = None,
        scale_constraint: Optional[Interval] = None,
    ) -> None:
        super().__init__(scale=scale)

        if scale_constraint is None:
            scale_constraint = Positive()

        self.raw_scale = torch.nn.Parameter(torch.ones(*batch_shape, 1, num_tasks) * scale)
        if scale_prior is not None:
            self.register_prior("scale_prior", scale_prior, lambda m: m.scale, lambda m, v: m._set_scale(v))

        self.register_constraint("raw_scale", scale_constraint)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        ret = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        
        num_event_dim = len(function_dist.event_shape)
        
        if num_event_dim > 1:  # Do appropriate summation for multitask likelihood
            ret = ret.sum(list(range(-1, -num_event_dim, -1)))
        return ret

class HadamardBetaLikelihood(MultitaskBetaLikelihood):
    
    def forward(self, function_samples, *args, **kwargs):
        assert 'task_indices' in kwargs.keys(), 'task_indices must be passed as a keyword argument'
        mixture = torch.distributions.Normal(0, 1).cdf(function_samples)

        task_indices = kwargs['task_indices']
        
        if self.scale.shape[-1]> 1:
            alpha_mask = torch.zeros_like(mixture)
            beta_mask = torch.zeros_like(mixture)
        
            for idx in torch.unique(task_indices):
                alpha_mask[:,task_indices == idx] = self.scale[:,idx] * mixture[:,task_indices == idx]
                beta_mask[:,task_indices == idx] = self.scale[:,idx] - alpha_mask[:,task_indices == idx]
            
            self.alpha = alpha_mask
            self.beta = beta_mask
        else:
            self.alpha = self.scale * mixture
            self.beta = self.scale - self.alpha
        
        self.alpha = torch.clamp(self.alpha, 1e-10, 1e10)
        self.beta = torch.clamp(self.beta, 1e-10, 1e10)

        return base_distributions.Beta(concentration1=self.alpha, concentration0=self.beta)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples, *args, **kwargs).log_prob(observations)
        ret = self.quadrature(log_prob_lambda, function_dist)
        
        num_event_dim = len(function_dist.event_shape)
        
        if num_event_dim > 1:  # Do appropriate summation for multitask likelihood
            ret = ret.sum(list(range(-1, -num_event_dim, -1)))
        return ret