import torch
import gpytorch
import numpy as np
from torch import nn
from torch.nn import Parameter
from torchdiffeq import odeint
from gpytorch.constraints import Interval, Positive
from gpytorch.metrics import negative_log_predictive_density as nlpd
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from models.gp_lfm import ExactGP, ApproximateGP

from typing import Optional


def discretisation_length(N, d):
    """Returns the length of a linspace where there are N points with d intermediate (in-between) points."""
    return (N - 1) * (d + 1) + 1

class ApproximatePVLFM(nn.Module):
    def __init__(self, dataset, num_outputs, gp_model, config):
        super().__init__()
        
        self.config = config
        pv_init = dataset.pv_init
        self.dataset = dataset
        self.gp_model = gp_model
        
        self.raw_initial = torch.tensor(pv_init, dtype=torch.float64).repeat(num_outputs, 1)
        self.a_constr = Interval(0.0001, 1.0)
        self.b_constr = Interval(0.001, 1.0)
        self.c_constr = Interval(0.001, 1.0)
        
        self.positive_constr = Positive()
        
        self.raw_a = Parameter(self.a_constr.inverse_transform(0.005 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))
        self.raw_b = Parameter(self.b_constr.inverse_transform(0.3 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))
        self.raw_c = Parameter(self.c_constr.inverse_transform(0.15 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))
        self.raw_noise = Parameter(self.positive_constr.inverse_transform(0.1 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))

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
    
    def initial_state(self):
        return self.raw_initial
    
    @property
    def noise_rate(self):
        return self.positive_constr.transform(self.raw_noise)
    
    @noise_rate.setter
    def noise_rate(self, value):
        self.raw_noise = self.positive_constr.inverse_transform(value)
    
    def pre_train_gp(self, n_iter =100, lr=0.1, verbose=False):
        self.gp_model.fit(n_iter=n_iter, lr=lr, verbose=verbose)
        # release the gp model from computation graph
        self.gp_model.eval()
    
    def odefunc(self, t, h):

        f = self.f[:, :, self.t_index].unsqueeze(2)
        dh = -self.a_rate * f  + self.b_rate * torch.sin(self.c_rate * t * np.pi) * h
      
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
        
        # self.gp_model.eval()
        qf = self.gp_model.likelihood(self.gp_model(t_f)).rsample(torch.Size([self.config.num_samples]))
            
        self.f = qf.unsqueeze(1) if len(qf.shape) == 2 else qf

        h0 = self.initial_state()
        h0 = h0.repeat(self.config.num_samples, 1, 1)
        h_samples_noiseless = odeint(self.odefunc, h0, t_f, method='rk4', options=dict(step_size=1))
        
        h_mean = torch.mean(h_samples_noiseless, dim=1).squeeze(-1).squeeze(-1)
        
        noise = torch.rand_like(h_samples_noiseless) * self.noise_rate
        h_samples = h_samples_noiseless + noise
       
        h_var = torch.var(h_samples, dim=1).transpose(0, 1).squeeze(-1).squeeze(-1) + 1e-6
        h_covar = torch.diag_embed(h_var)
        h_dist = gpytorch.distributions.MultivariateNormal(h_mean, h_covar)

        return h_dist
    
    def predict_pv(self, 
                   t, 
                   initial_state_pred, 
                   initial_var_pred,
                   step_size=0.33, 
                   num_discrete : Optional[int] = None,
                   fit_gp=False):
        # get gp samples at t
        if fit_gp:
            self.gp_model = ApproximateGP(
                self.dataset.data_test,
                ZeroMean(),
                ScaleKernel(RBFKernel()),
                gpytorch.likelihoods.GaussianLikelihood(),
                config= {
                    'type' : 'stochastic',
                    'name' : 'cholesky',
                    'jitter' : 1e-6,
                    'num_inducing_points' : self.dataset.times_test.shape[0]
                }
            )
            self.gp_model.fit(n_iter=200, lr=0.1, verbose=False)
            
        self.gp_model.eval()
        qf = self.gp_model(t).sample(torch.Size([self.config.num_samples]))
        self.f = qf.unsqueeze(1) if len(qf.shape) == 2 else qf
  
        h0 = torch.distributions.Normal(initial_state_pred, initial_var_pred).sample(torch.Size([self.config.num_samples])).unsqueeze(-1).unsqueeze(-1)
        h0 = torch.clamp(h0, 0, 1e10)

        self.t_index = 0
        self.last_t = -1

        if num_discrete is None:
            t_f = t
            step_size = 1
        else:
            t_f = torch.linspace(t.min(), t.max(), discretisation_length(t.shape[0], num_discrete))
        
        h_samples_noiseless = odeint(self.odefunc, h0, t_f, method='rk4', options=dict(step_size=step_size))        
        h_mean = torch.mean(h_samples_noiseless, dim=1).squeeze(-1).squeeze(-1)
        
        noise = torch.rand_like(h_samples_noiseless) * self.noise_rate
        h_samples = h_samples_noiseless + noise
        
        h_covar = torch.var(h_samples, dim=1).transpose(0, 1).squeeze(-1).squeeze(-1) + 1e-6
        h_dist = gpytorch.distributions.MultivariateNormal(h_mean, torch.diag_embed(h_covar))

        with torch.no_grad():
            self.gp_model.eval()
            self.gp_model.likelihood.eval()
            latent_posterior = self.gp_model.likelihood(self.gp_model(t))
        
        return h_dist, latent_posterior
    
    def fit(self, n_iter=100, lr=0.02, step_size=0.33, warm_start=True):
        
        if warm_start:
            self.pre_train_gp(n_iter=n_iter // 3, lr=0.1, verbose=True)
        
        
        y_target = torch.tensor(self.dataset.pv, dtype=torch.float64)

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
            output_gp = self.gp_model(self.dataset.data[0][0])
            
            loss = nlpd(output_lfm, y_target) - elbo(output_gp, self.dataset.data[0][1])
            loss.backward()

            [opt.step() for opt in optimizer]

            if i % 10 == 0:
                print(f'Iter {i+1}/{n_iter}, Loss: {loss.item()}')
        
        self.params = {
            'a': self.a_rate.item(),
            'b': self.b_rate.item(),
            'c': self.c_rate.item(),
        }

class ExactPVLFM(nn.Module):
    def __init__(self, dataset, num_outputs, gp_model, config):
        super().__init__()
        
        self.config = config
        pv_init = dataset.pv_init
        self.dataset = dataset
        self.gp_model = gp_model
        
        self.raw_initial = torch.tensor(pv_init, dtype=torch.float64).repeat(num_outputs, 1)
        self.a_constr = Interval(0.0001, 1.0)
        self.b_constr = Interval(0.001, 1.0)
        self.c_constr = Interval(0.001, 1.0)
        
        self.positive_constr = Positive()
        
        self.raw_a = Parameter(self.a_constr.inverse_transform(0.005 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))
        self.raw_b = Parameter(self.b_constr.inverse_transform(0.3 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))
        self.raw_c = Parameter(self.c_constr.inverse_transform(0.15 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))
        self.raw_noise = Parameter(self.positive_constr.inverse_transform(0.1 * torch.ones(torch.Size([num_outputs, 1]), dtype=torch.float64)))


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
    
    def initial_state(self):
        return self.raw_initial
    
    @property
    def noise_rate(self):
        return self.positive_constr.transform(self.raw_noise)
    
    @noise_rate.setter
    def noise_rate(self, value):
        self.raw_noise = self.positive_constr.inverse_transform(value)
    
    def pre_train_gp(self, n_iter =100, lr=0.1, verbose=False):
        self.gp_model.fit(n_iter=n_iter, lr=lr, verbose=verbose)
    
    def odefunc(self, t, h):

        f = self.f[:, :, self.t_index].unsqueeze(2)
        dh = -self.a_rate * f  + self.b_rate * torch.sin(self.c_rate * t * np.pi) * h
      
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
        
        self.gp_model.eval()
        qf = self.gp_model.likelihood(self.gp_model(t_f)).rsample(torch.Size([self.config.num_samples]))
            
        self.f = qf.unsqueeze(1) if len(qf.shape) == 2 else qf

        h0 = self.initial_state()
        h0 = h0.repeat(self.config.num_samples, 1, 1)
        h_samples_noiseless = odeint(self.odefunc, h0, t_f, method='rk4', options=dict(step_size=1))
        
        h_mean = torch.mean(h_samples_noiseless, dim=1).squeeze(-1).squeeze(-1)
        
        noise = torch.rand_like(h_samples_noiseless) * self.noise_rate
        h_samples = h_samples_noiseless + noise
       
        h_var = torch.var(h_samples, dim=1).transpose(0, 1).squeeze(-1).squeeze(-1) + 1e-6
        h_covar = torch.diag_embed(h_var)
        h_dist = gpytorch.distributions.MultivariateNormal(h_mean, h_covar)

        return h_dist
    
    def predict_pv(self, 
                   t, 
                   initial_state_pred, 
                   initial_var_pred,
                   step_size=0.33, 
                   num_discrete : Optional[int] = None,
                   fit_gp=False):
        # get gp samples at t
        if fit_gp:
            self.gp_model = ExactGP(
                self.dataset.data_test[0][0],
                self.dataset.data_test[0][1],
                self.gp_model.mean_module,
                self.gp_model.covar_module,
                self.gp_model.likelihood
            )
            self.gp_model.fit(n_iter=1000, lr=0.1, verbose=False)
            
        self.gp_model.eval()
        qf = self.gp_model(t).sample(torch.Size([self.config.num_samples]))
        self.f = qf.unsqueeze(1) if len(qf.shape) == 2 else qf
  
        h0 = torch.distributions.Normal(initial_state_pred, initial_var_pred).sample(torch.Size([self.config.num_samples])).unsqueeze(-1).unsqueeze(-1)
        h0 = torch.clamp(h0, 0, 1e10)

        self.t_index = 0
        self.last_t = -1

        if num_discrete is None:
            t_f = t
            step_size = 1
        else:
            t_f = torch.linspace(t.min(), t.max(), discretisation_length(t.shape[0], num_discrete))
        
        h_samples_noiseless = odeint(self.odefunc, h0, t_f, method='rk4', options=dict(step_size=step_size))        
        h_mean = torch.mean(h_samples_noiseless, dim=1).squeeze(-1).squeeze(-1)
        
        noise = torch.rand_like(h_samples_noiseless) * self.noise_rate
        h_samples = h_samples_noiseless + noise
        
        h_covar = torch.var(h_samples, dim=1).transpose(0, 1).squeeze(-1).squeeze(-1) + 1e-6
        h_dist = gpytorch.distributions.MultivariateNormal(h_mean, torch.diag_embed(h_covar))

        with torch.no_grad():
            self.gp_model.eval()
            self.gp_model.likelihood.eval()
            latent_posterior = self.gp_model.likelihood(self.gp_model(t))
        
        return h_dist, latent_posterior

    def fit(self, n_iter=100, lr=0.02, step_size=0.33, warm_start=True):
        
        if warm_start:
            self.pre_train_gp(n_iter=100, lr=0.1, verbose=False)
        
        y_target = torch.tensor(self.dataset.pv, dtype=torch.float64)
        lfm_params = [p for p in self.parameters() if p not in self.gp_model.parameters()]
        optimizer = torch.optim.Adam(lfm_params, lr=lr)
        
        
        for i in range(n_iter):
            optimizer.zero_grad()
            output_lfm = self(self.dataset.times, step_size=step_size, idx=i)
            
            loss = nlpd(output_lfm, y_target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'Iteration {i}, loss {loss.item()}')
        
        self.params = {
            'a': self.a_rate.item(),
            'b': self.b_rate.item(),
            'c': self.c_rate.item(),
           
        }