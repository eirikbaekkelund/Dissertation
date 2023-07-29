import torch
import gpytorch
import numpy as np
from gpytorch.distributions import MultivariateNormal
from alfi.models.lfm import LFM
from alfi.means import SIMMean
from kernels.sim import SIMKernel
from alfi.datasets import LFMDataset
from alfi.utilities.data import flatten_dataset


class ExactLFM(LFM, gpytorch.models.ExactGP):
    """
    An implementation of the single input motif from Lawrence et al., 2006.
    """
    def __init__(self, dataset: LFMDataset, variance):
        train_t, train_y = flatten_dataset(dataset)

        super().__init__(train_t, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())

        self.num_outputs = dataset.num_outputs
     
        self.train_t = train_t
        self.train_y = train_y
       # TODO consider if variance should have a gradient
        self.covar_module = SIMKernel(self.num_outputs, torch.tensor(variance, requires_grad=False))
        initial_basal = torch.mean(train_y.view(self.num_outputs, -1), dim=1) * self.covar_module.decay
        self.mean_module = SIMMean(self.covar_module, self.num_outputs, initial_basal)

    @property
    def decay_rate(self):
        return self.covar_module.decay

    @decay_rate.setter
    def decay_rate(self, val):
        self.covar_module.decay = val

    def forward(self, x):

        x.flatten()
        # TODO may remove probit transform of mean
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) 

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict_m(self, pred_t, jitter=1e-3) -> torch.distributions.MultivariateNormal:
        """
        Predict outputs of the LFM
        
        Args:
            pred_t (torch.Tensor): Prediction times
            jitter (float, optional): Jitter to add to the diagonal of the covariance matrix. Defaults to 1e-3.
        
        Returns:
            torch.distributions.MultivariateNormal: Predicted outputs
        """
        pred_t_blocked = pred_t.repeat(self.num_outputs)
        
        Kxx = self.covar_module(self.train_t, self.train_t)
        K_inv = torch.inverse(Kxx.evaluate())
                
        K_xxstar = self.covar_module(self.train_t, pred_t_blocked).evaluate()
        K_xstarx = torch.transpose(K_xxstar, 0, 1).type(torch.float64)
        
        K_xstarxK_inv = torch.matmul(K_xstarx, K_inv)
        KxstarxKinvY = torch.matmul(K_xstarxK_inv.double(), self.train_y.double())
        K_xstarxstar = self.covar_module(pred_t_blocked, pred_t_blocked).evaluate()
        
        var = K_xstarxstar - torch.matmul(K_xstarxK_inv, torch.transpose(K_xstarx, 0, 1))
        
        var = torch.diagonal(var, dim1=0, dim2=1).view(self.num_outputs, pred_t.shape[0])
        var = var.transpose(0, 1)
        var = torch.diag_embed(var)
        var += jitter * torch.eye(var.shape[-1])
        
        mean = KxstarxKinvY.view(self.num_outputs, pred_t.shape[0])
        mean = mean.transpose(0, 1)
        


        return MultivariateNormal(mean, var)

    def predict_f(self, pred_t, jitter=1e-5) -> MultivariateNormal:
        """
        Predict the latent function.

        Args:
            pred_t (torch.Tensor): Prediction times
            jitter (float, optional): Jitter to add to the diagonal of the covariance matrix. Defaults to 1e-5.
        
        Returns:
            MultivariateNormal: Predicted latent function
        """
        Kxx = self.covar_module(self.train_t, self.train_t)
        K_inv = torch.inverse(Kxx.evaluate())

        Kxf = self.covar_module.K_xf(self.train_t, pred_t).type(torch.float64)
        KfxKxx = torch.matmul(torch.transpose(Kxf, 0, 1), K_inv)
        mean = torch.matmul(KfxKxx.double(), self.train_y.double()).view(-1).unsqueeze(0)

        #Kff-KfxKxxKxf
        Kff = self.covar_module.K_ff(pred_t, pred_t)  # (100, 500)
        var = Kff - torch.matmul(KfxKxx, Kxf)
        # var = torch.diagonal(var, dim1=0, dim2=1).view(-1)
        var = var.unsqueeze(0)
        # For some reason a full covariance is not PSD, for now just take the variance: (TODO)
        var = torch.diagonal(var, dim1=1, dim2=2)
        var = torch.diag_embed(var)
        var += jitter * torch.eye(var.shape[-1])

        batch_mvn = gpytorch.distributions.MultivariateNormal(mean, var)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath+'lfm.pt')

    @classmethod
    def load(cls,
             filepath,
             lfm_args=[], lfm_kwargs={}):
        lfm_state_dict = torch.load(filepath+'lfm.pt')
        lfm = cls(*lfm_args, **lfm_kwargs)
        lfm.load_state_dict(lfm_state_dict)
        return lfm