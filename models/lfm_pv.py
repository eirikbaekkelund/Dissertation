import torch
import gpytorch
import numpy as np
from torch import nn
from torch.nn import Parameter
from torchdiffeq import odeint
from gpytorch.constraints import Interval
from typing import Optional


def discretisation_length(N, d):
    """Returns the length of a linspace where there are N points with d intermediate (in-between) points."""
    return (N - 1) * (d + 1) + 1

class ApproximatePVLFM(nn.Module):
    """ 
    Outputs are cloud cover, latent is PV
    """
    def __init__(self, dataset, num_outputs, gp_model, config):
        super().__init__()
        
        self.config = config
        init = dataset.cloud_init
        self.dataset = dataset
        self.gp_model = gp_model
        
        self.raw_initial = torch.tensor(init, dtype=torch.float64).repeat(num_outputs, 1)
        self.a_constr = Interval(0.0001, 1.0)
        self.b_constr = Interval(0.001, 1.0)
        self.c_constr = Interval(0.001, 1.0)
        self.noise_constr = Interval(0.05, 0.5)
        
        self.raw_a = Parameter(self.a_constr.inverse_transform(0.005 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))
        self.raw_b = Parameter(self.b_constr.inverse_transform(0.3 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))
        self.raw_c = Parameter(self.c_constr.inverse_transform(0.2 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))
        self.raw_noise = Parameter(self.noise_constr.inverse_transform(0.2 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))

    @property
    def a_rate(self):
        return self.a_constr.transform(self.raw_a)
    
    @a_rate.setter
    def a_rate(self, value):
        self.raw_a = self.a_constr.inverse_transform(value)
    
    @property
    def b_rate(self):
        return self.b_constr.transform(self.raw_b)
    
    @b_rate.setter
    def b_rate(self, value):
        self.raw_b = self.b_constr.inverse_transform(value)
    
    @property
    def c_rate(self):
        return self.c_constr.transform(self.raw_c)
    
    @c_rate.setter
    def c_rate(self, value):
        self.raw_c = self.c_constr.inverse_transform(value)
    
    @property
    def initial(self):
        return self.raw_initial
    
    @initial.setter
    def initial(self, value):
        self.raw_initial = value
    
    @property
    def noise_rate(self):
        return self.noise_constr.transform(self.raw_noise)
    
    @noise_rate.setter
    def noise_rate(self, value):
        self.raw_noise = self.noise_constr.inverse_transform(value)
    
    def initial_state(self):
        return self.raw_initial
    
    def G(self, f):
        return f
    
    def odefunc(self, t, h):

        f = self.G(self.f[:, :, self.t_index].unsqueeze(2))
        dh = self.a_rate * f  - self.b_rate * torch.sin(self.c_rate * t * np.pi) * h
      
        if t > self.last_t:
            self.t_index += 1
        self.last_t = t

        return dh
    
    def forward(self, t, step_size=0.33, num_discrete : Optional[list] = None, idx : Optional[int] = None):
        
        if num_discrete is None:
            t_f = t
            step_size = 1
        else:
            t_f = torch.linspace(t.min(), t.max(), discretisation_length(t.shape[0], num_discrete))
        
        self.t_index = 0
        self.last_t = -1
        qf = self.gp_model(t_f).rsample(torch.Size([self.config.num_samples]))
        self.f = qf.unsqueeze(1) if len(qf.shape) == 2 else qf

        h0 = self.initial_state()
        h0 = h0.repeat(self.config.num_samples, 1, 1)
        h_samples_noiseless = odeint(self.odefunc, h0, t_f, method='rk4', options=dict(step_size=step_size))
        
        h_mean = torch.mean(h_samples_noiseless, dim=1).squeeze(-1).squeeze(-1)

        # small noise added to the cloud cover output as in the data
        h_samples = h_samples_noiseless + torch.randn_like(h_samples_noiseless) * self.noise_rate
        
        h_var = torch.var(h_samples, dim=1).transpose(0, 1).squeeze(-1).squeeze(-1)
        
        h_covar = torch.diag_embed(h_var) + 1e-4 * torch.eye(h_var.shape[0], dtype=torch.float64)
        h_dist = gpytorch.distributions.MultivariateNormal(h_mean, h_covar)
        
        return h_dist
    
    def predict_pv(self, initial_pv, var_pv, t, step_size=0.33, num_discrete : Optional[list] = None, idx : Optional[int] = None):
        """ 
        Predict latent PV given the cloud cover.
        """
        def ode_latent(t, h):
            f = self.G(self.f[:, :, self.t_index].unsqueeze(2))
            dh = -self.a_rate * f  + self.b_rate * torch.sin(self.c_rate * t * np.pi) * h
            if t > self.last_t:
                self.t_index += 1
            self.last_t = t
            
            return dh
        
        # get prediction of cloud cover as our probabilistic input to the ODE for PV
        dist = self(t, step_size=step_size, num_discrete=num_discrete, idx=idx)
        qf = dist.sample(torch.Size([self.config.num_samples]))
        self.f = qf.unsqueeze(1) if len(qf.shape) == 2 else qf

        h0 = torch.distributions.Normal(initial_pv, var_pv).sample(torch.Size([self.config.num_samples])).unsqueeze(-1).unsqueeze(-1)
        h0 = torch.clamp(h0, 0, 1e10)

        self.t_index = 0
        self.last_t = -1

        h_samples = odeint(ode_latent, h0, t, method='rk4', options=dict(step_size=1))

        h_mean = torch.mean(h_samples, dim=1).squeeze(-1).squeeze(-1)
        h_var = torch.var(h_samples, dim=1).transpose(0, 1).squeeze(-1).squeeze(-1) + var_pv

        h_covar = torch.diag_embed(h_var) + 1e-4 * torch.eye(h_var.shape[0], dtype=torch.float64)
        h_dist = gpytorch.distributions.MultivariateNormal(h_mean.detach(), h_covar.detach())

        return h_dist
    
    def predict_cloud(self, initial_cloud, var_cloud, t):
        """ 
        Predict cloud cover given the latent PV.
        """
        def ode_cloud(t, h):
            f = self.G(self.f[:, :, self.t_index].unsqueeze(2))
            dh = self.a_rate * f  - self.b_rate * torch.sin(self.c_rate * t * np.pi) * h
            if t > self.last_t:
                self.t_index += 1
            self.last_t = t
            
            return dh
        
        # get prediction of cloud cover as our probabilistic input to the ODE for PV
        self.t_index = 0
        self.last_t = -1
        qf = self.gp_model(t).sample(torch.Size([self.config.num_samples]))
        self.f = qf.unsqueeze(1) if len(qf.shape) == 2 else qf

        h0 = torch.distributions.Normal(initial_cloud, var_cloud).sample(torch.Size([self.config.num_samples])).unsqueeze(-1).unsqueeze(-1)
        h0 = torch.clamp(h0, 0, 1e10)


        h_samples = odeint(ode_cloud, h0, t, method='rk4', options=dict(step_size=1))

        h_mean = torch.mean(h_samples, dim=1).squeeze(-1).squeeze(-1)
        h_var = torch.var(h_samples, dim=1).transpose(0, 1).squeeze(-1).squeeze(-1) + var_cloud

        h_covar = torch.diag_embed(h_var) + 1e-4 * torch.eye(h_var.shape[0], dtype=torch.float64)
        h_dist = gpytorch.distributions.MultivariateNormal(h_mean.detach(), h_covar.detach())

        return h_dist


    def fit(self, n_iter=100, lr=0.02, step_size=0.33):
        
        lfm_params = [param for name, param in self.named_parameters() if 'gp_model' not in name]
        gp_params = [param for name, param in self.named_parameters() if 'gp_model' in name]

        optimizer = [torch.optim.Adam(lfm_params, lr=lr),
                     torch.optim.Adam(gp_params, lr=lr)]
        
        self.train()
        self.gp_model.train()
        self.gp_model.likelihood.train()
        
        elbo = gpytorch.mlls.VariationalELBO(self.gp_model.likelihood, self.gp_model, num_data=len(self.dataset.data[0][0]))

        for i in range(n_iter):
            [opt.zero_grad() for opt in optimizer]
            output_lfm = self(self.dataset.times, step_size=step_size, idx=i)
            loss = -elbo(output_lfm, self.dataset.data[0][1]) 
            loss.backward()

            [opt.step() for opt in optimizer]

            if (i + 1) % 10 == 0:
                print(f'Iter {i+1}/{n_iter}, Loss: {loss.item()}')
        
            self.params = {
                'a': self.a_rate.item(),
                'b': self.b_rate.item(),
                'c': self.c_rate.item(),
            }
