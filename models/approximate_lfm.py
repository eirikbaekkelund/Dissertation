from alfi.models import LFM
from abc import ABC, abstractmethod
import torch
import gpytorch
from torchdiffeq import odeint
from alfi.configuration import VariationalConfiguration
from alfi.utilities.torch import is_cuda
from alfi.models.lfm import LFM
from alfi.mlls import MaskedVariationalELBO
from alfi.plot import Plotter1d, plot_phase, Colours
from alfi.utilities.torch import softplus, inv_softplus
from torch.nn.parameter import Parameter
from gpytorch.models import ApproximateGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.constraints import Positive, Interval
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    NaturalVariationalDistribution,
    CholeskyVariationalDistribution,
    VariationalStrategy,
    IndependentMultitaskVariationalStrategy,
    TrilNaturalVariationalDistribution
)

class MultiOutputGP(ApproximateGP):
    def __init__(self,
                 mean_module,
                 covar_module,
                 inducing_points,
                 num_latents,
                 learn_inducing_locations=False,
                 natural=True,
                 use_tril=False):
        # The variational dist batch shape means we learn a different variational dist for each latent
        if natural:
            Distribution = TrilNaturalVariationalDistribution if use_tril else NaturalVariationalDistribution
            variational_distribution = Distribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )
        else:
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )

        # Wrap the VariationalStrategy in a MultiTask to make output MultitaskMultivariateNormal
        # rather than a batch MVN
        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_locations
            ), num_tasks=num_latents
        )
        super().__init__(variational_strategy)

        self.mean_module = mean_module
        self.covar_module = covar_module

    def get_inducing_points(self):
        return self.variational_strategy.base_variational_strategy.inducing_points

    def forward(self, t):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(t)
        covar_x = self.covar_module(t)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def generate_multioutput_rbf_gp(num_latents, inducing_points,
                                ard_dims=None,
                                use_scale=False,
                                initial_lengthscale=None,
                                lengthscale_constraint=None,
                                zero_mean=True,
                                gp_kwargs={}):
    # Modules should be marked as batch so different set of hyperparameters are learnt
    if zero_mean:
        mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
    else:
        mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
    covar_module = gpytorch.kernels.RBFKernel(
        batch_shape=torch.Size([num_latents]),
        ard_num_dims=ard_dims,
        lengthscale_constraint=lengthscale_constraint
    )
    if use_scale:
        covar_module = gpytorch.kernels.ScaleKernel(
            covar_module,
            batch_shape=torch.Size([num_latents])
        )
    if initial_lengthscale is not None:
        if use_scale:
            covar_module.base_kernel.lengthscale = initial_lengthscale
        else:
            covar_module.lengthscale = initial_lengthscale
    return MultiOutputGP(mean_module, covar_module, inducing_points, num_latents, **gp_kwargs)

class VariationalLFM(LFM, ABC):
    """
    Variational inducing point approximation for Latent Force Models.

    Parameters
    ----------
    num_outputs : int : the number of outputs (for example, the number of genes)
    fixed_variance : tensor : variance if the preprocessing variance is known, otherwise learnt.
    """
    def __init__(self,
                 num_outputs: int,
                 gp_model: ApproximateGP,
                 config: VariationalConfiguration,
                 num_training_points=None,
                 dtype=torch.float64):
        super().__init__()
        self.gp_model = gp_model
        self.num_outputs = num_outputs
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=self.num_outputs)
        self.pretrain_mode = False
        try:
            self.inducing_points = self.gp_model.get_inducing_points()
        except AttributeError:
            raise AttributeError('The GP model must define a function `get_inducing_points`.')

        if num_training_points is None:
            num_training_points = self.inducing_points.numel()  # TODO num_data refers to the number of training datapoints

        self.loss_fn = MaskedVariationalELBO(self.likelihood, gp_model, num_training_points, combine_terms=False)
        self.config = config
        self.dtype = dtype

        # if config.preprocessing_variance is not None:
        #     self.likelihood_variance = Parameter(torch.tensor(config.preprocessing_variance), requires_grad=False)
        # else:
        #     self.raw_likelihood_variance = Parameter(torch.ones((self.num_outputs, self.num_observed), dtype=dtype))

        if config.initial_conditions:
            self.initial_conditions = Parameter(torch.tensor(torch.zeros(self.num_outputs, 1)), requires_grad=True)

    def nonvariational_parameters(self):
        variational_keys = dict(self.gp_model.named_variational_parameters()).keys()
        named_parameters = dict(self.named_parameters())
        return [named_parameters[key] for key in named_parameters.keys()
                if key[len('gp_model.'):] not in variational_keys]

    def variational_parameters(self):
        return self.gp_model.variational_parameters()

    def summarise_gp_hyp(self):
        # variational_keys = dict(self.gp_model.named_variational_parameters()).keys()
        # named_parameters = dict(self.named_parameters())
        #
        # return [named_parameters[key] for key in named_parameters.keys()
        #         if key[len('gp_model.'):] not in variational_keys]
        if self.gp_model.covar_module.lengthscale is not None:
            return self.gp_model.covar_module.lengthscale.detach().cpu().numpy()
        elif hasattr(self.gp_model.covar_module, 'base_kernel'):
            kernel = self.gp_model.covar_module.base_kernel
            if hasattr(kernel, 'kernels'):
                if hasattr(kernel.kernels[0], 'lengthscale'):
                    return kernel.kernels[0].lengthscale.detach().cpu().numpy()
            else:
                return self.gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        else:
            return ''

    def forward(self, x):
        raise NotImplementedError

    def train(self, mode: bool = True):
        self.gp_model.train(mode)
        self.likelihood.train(mode)

    def pretrain(self, mode=True):
        self.pretrain_mode = mode

    def eval(self):
        self.gp_model.eval()
        self.likelihood.eval()
        self.pretrain(False)

    def predict_m(self, t_predict, **kwargs) -> torch.distributions.MultivariateNormal:
        """
        Calls self on input `t_predict`
        """
        return self(t_predict.view(-1), **kwargs)

    def predict_f(self, t_predict, **kwargs) -> torch.distributions.MultivariateNormal:
        """
        Returns the latents
        """
        self.eval()
        with torch.no_grad():
            q_f = self.gp_model(t_predict)
        self.train()
        return q_f

    def save(self, filepath):
        torch.save(self.gp_model.state_dict(), str(filepath)+'gp.pt')
        torch.save(self.state_dict(), str(filepath)+'lfm.pt')

    @classmethod
    def load(cls,
             filepath,
             gp_cls=None, gp_model=None,
             gp_args=[], gp_kwargs={},
             lfm_args=[], lfm_kwargs={}):
        assert not (gp_cls is None and (gp_model is None))
        gp_state_dict = torch.load(filepath+'gp.pt')
        if gp_cls is not None:
            gp_model = gp_cls(*gp_args, **gp_kwargs)
        gp_model.load_state_dict(gp_state_dict)
        gp_model.double()

        lfm_state_dict = torch.load(filepath+'lfm.pt')
        lfm = cls(lfm_args[0], gp_model, *lfm_args[1:], **lfm_kwargs)
        lfm.load_state_dict(lfm_state_dict)
        return lfm
    
class OrdinaryLFM(VariationalLFM):
    """
    Variational approximation for an LFM based on an ordinary differential equation (ODE).
    Inheriting classes must override the `odefunc` function which encodes the ODE.
    """

    def __init__(self, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
        super().__init__(num_outputs, gp_model, config, **kwargs)
        self.nfe = 0
        self.f = None

    def initial_state(self):
        initial_state = torch.zeros(torch.Size([self.num_outputs, 1]), dtype=torch.float64)
        initial_state = initial_state.cuda() if is_cuda() else initial_state
        return initial_state #initial_state.repeat(self.config.num_samples, 1, 1)  # Add batch dimension for sampling
        # if self.config.initial_conditions: TODO:
        #     h = self.initial_conditions.repeat(h.shape[0], 1, 1)

    def forward(self, t, step_size=1e-1, return_samples=False, **kwargs):
        """
        t : torch.Tensor
            Shape (num_times)
        h : torch.Tensor the initial state of the ODE
            Shape (num_genes, 1)
        Returns
        -------
        Returns evolved h across times t.
        Shape (num_genes, num_points).
        """
        self.nfe = 0

        # Get GP outputs
        if self.pretrain_mode:
            t_f = t[0]
            h0 = t[1]
        else:
            t_f = torch.arange(t.min(), t.max()+ 2*step_size/3, step_size/3)
            h0 = self.initial_state()
            h0 = h0.unsqueeze(0).repeat(self.config.num_samples, 1, 1)

        q_f = self.gp_model(t_f)

        self.f = q_f.rsample(torch.Size([self.config.num_samples])).permute(0, 2, 1)  # (S, I, T)
        self.f = self.G(self.f)

        if self.pretrain_mode:
            h_samples = self.odefunc(t_f, h0)
            h_samples = h_samples.permute(2, 0, 1)
        else:
            # Integrate forward from the initial positions h0.
            self.t_index = 0
            self.last_t = self.f.min() - 1
            h_samples = odeint(self.odefunc, h0, t, method='rk4', options=dict(step_size=step_size)) # (T, S, num_outputs, 1)

        self.f = None
        # self.t_index = None
        # self.last_t = None
        if return_samples:
            return h_samples

        h_mean = torch.mean(h_samples, dim=1).squeeze(-1).transpose(0, 1)  # shape was (#outputs, #T, 1)
        h_var = torch.var(h_samples, dim=1).squeeze(-1).transpose(0, 1) + 1e-7
        h_mean = self.decode(h_mean)
        h_var = self.decode(h_var)
        # TODO: make distribution something less constraining
        h_covar = torch.diag_embed(h_var) + torch.eye(self.num_outputs, dtype=torch.float64) * 1e-1
        batch_mvn = gpytorch.distributions.MultivariateNormal(h_mean, h_covar)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(batch_mvn, task_dim=0)

    def decode(self, h_out):
        return h_out

    @abstractmethod
    def odefunc(self, t, h):
        """
        Parameters:
            h: shape (num_samples, num_outputs, 1)
        """
        pass

    def G(self, f):
        return f.repeat(1, self.num_outputs, 1)  # (S, I, t)


class LotkaVolterra(OrdinaryLFM):
    """Outputs are predator. Latents are prey"""
    def __init__(self, num_outputs, gp_model, config: VariationalConfiguration, **kwargs):
        super().__init__(num_outputs, gp_model, config, **kwargs)
        self.num_tasks = num_outputs
        self.positivity = Positive()
        self.decay_constraint = Interval(0., 100.)
        self.raw_decay = Parameter(self.decay_constraint.inverse_transform(torch.ones(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
        self.raw_growth = Parameter(self.positivity.inverse_transform(0.5*torch.ones(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))
        self.raw_initial = Parameter(self.positivity.inverse_transform(0.3+torch.zeros(torch.Size([self.num_outputs, 1]), dtype=torch.float64)))

    @property
    def decay_rate(self):
        return self.decay_constraint.transform(self.raw_decay)

    @decay_rate.setter
    def decay_rate(self, value):
        self.raw_decay = self.decay_constraint.inverse_transform(value)

    @property
    def growth_rate(self):
        return softplus(self.raw_growth)

    @growth_rate.setter
    def growth_rate(self, value):
        self.raw_growth = inv_softplus(value)

    @property
    def initial_predators(self):
        return softplus(self.raw_initial)

    @initial_predators.setter
    def initial_predators(self, value):
        self.raw_initial = inv_softplus(value)

    def initial_state(self):
        return self.initial_predators

    def odefunc(self, t, h):
        """h is of shape (num_samples, num_outputs, 1)"""
        self.nfe += 1
        
        f = self.G(self.f[:, :, self.t_index].unsqueeze(2))
        dh = self.growth_rate * h * f - self.decay_rate * h
        
        if t > self.last_t:
            self.t_index += 1
        self.last_t = t
        
        return dh

    def G(self, f):
        # I = 1 so just repeat for num_outputs
        #softplus
        return softplus(f).repeat(1, self.num_outputs, 1)