import numpy as np
import torch
from torchdiffeq import odeint
from alfi.utilities.torch import discretisation_length

class SyntheticPV:
    """ 
    Generate synthetic PV data using an ODE model.
    This will likely be inaccurate, but highlights the use of the LFM.

    It will be a 2D system with the following ODE (i.e. they are inverses of each other)
    dPV/dt = -a * C * PV + b *sin(c * t)
    dC/dt = a * C * PV - b * sin(c * t)

    """
    def __init__(self,
            pv_init : float = 0.3,
            cloud_init : float = 0.7,
            time_steps = 60,
            end_time = 60,
            num_discrete = 1,
            prop_train = 0.5,
            ):
        self.pv_init = pv_init
        self.cloud_init = cloud_init
        self.time_steps = time_steps
        self.end_time = end_time
        self.num_discrete = num_discrete

        times, X = self.generate_ts()

        self.data = []
        n_train = int(len(times) * prop_train)
        times_train = times[:n_train]
        X_train = X[:n_train]
        self.data.append((times_train, X_train[:, 1]))
        self.pv = X_train[:, 0]
        self.cloud = X_train[:, 1]
        self.times = times_train

        self.times_test = times[n_train:]
        self.data_test = []
        
        self.data_test.append((self.times_test, X[n_train:][:, 1]))
        self.pv_test = X[n_train:, 0]
        self.cloud_test = X[n_train:, 1]
    
    def generate_ts(self):
        a = np.random.uniform(0.001, 0.01)
        b = np.random.uniform(0.1, 0.5)
        c = np.random.uniform(0.1, 0.2)
      
        def dX_dt(t, X):
            return torch.stack([
               -a * X[1] + b * torch.sin(c * t * np.pi) *  X[0],
               a * X[0] - b * torch.sin(c * t * np.pi) * X[1],
               
            ])
        
        t = torch.linspace(0, self.end_time, discretisation_length(self.time_steps, self.num_discrete))
        X0 = torch.tensor([self.pv_init, self.cloud_init])
        X = odeint(dX_dt, X0, t, method='rk4', options=dict(step_size=5e-2))
        X += torch.randn_like(X) * 0.05
       

        self.true_params = {'a': a, 'b': b, 'c': c}
       
        return t, X
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)