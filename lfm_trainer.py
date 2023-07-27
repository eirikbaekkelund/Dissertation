import torch
import gpytorch
import numpy as np
import sys
from alfi.models import VariationalLFM
from alfi.utilities.torch import is_cuda
from typing import List

sys.path.append('.')

from alfi.trainers.torte import Trainer as TorteTrainer

class Trainer(TorteTrainer):
    def print_extra(self):
        if isinstance(self.model, gpytorch.models.GP):
            kernel = self.model.covar_module
            print(f'λ: {str(kernel.lengthscale.view(-1).detach().numpy())}', end='')
        elif hasattr(self.model, 'gp_model'):
            print(f'kernel: {self.model.summarise_gp_hyp()}', end='')
        super().print_extra()


class ExactTrainer(Trainer):
    def __init__(self, *args, loss_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.losses = np.empty((0, 1))

    def single_epoch(self, **kwargs):
        [optim.zero_grad() for optim in self.optimizers]
        
        output = self.model(self.model.train_t)        
        loss = -self.loss_fn(output, self.model.train_y) 
        loss.backward()
        
        [optim.step() for optim in self.optimizers]
        
        epoch_loss = loss.item()

        return epoch_loss, [epoch_loss]

    def print_extra(self):
        self.model.covar_module.lengthscale.item(),
        self.model.likelihood.noise.item()
        super().print_extra()

class VariationalTrainer(Trainer):
    """
    Parameters:
        batch_size: in the case of the transcriptional regulation model, we train the entire gene set as a batch
    """
    def __init__(self,
                 lfm: VariationalLFM,
                 optimizers: List[torch.optim.Optimizer],
                 dataset,
                 step_size,
                 warm_variational=-1,
                 **kwargs):
        super().__init__(lfm, optimizers, dataset, batch_size=lfm.num_tasks, **kwargs)
        self.warm_variational = warm_variational
        self.step_size = step_size
    
    def single_epoch(self, epoch=0, **kwargs):
        
        epoch_loss = 0
        epoch_ll = 0
        epoch_kl = 0
        
        for i, data in enumerate(self.data_loader):
        
            [optim.zero_grad() for optim in self.optimizers]
        
            data_input, y = data
            data_input = data_input.cuda() if is_cuda() else data_input
        
            y = y.cuda() if is_cuda() else y
        
            # Assume that the batch of t s are the same
            print('data_input', data_input.shape)
            print('y', y.shape)
        
            data_input = data_input.squeeze(0).float()
        
            output = self.model(data_input, step_size=self.step_size)
            y_target = y.t().squeeze(-1)
            
            print('output', output)
            print('y_target', y_target.shape)
            
            log_likelihood, kl_divergence, _ = self.model.loss_fn(output, y_target, mask=self.train_mask)

            loss = - (log_likelihood - kl_divergence)
            loss.backward()
            
            if epoch >= self.warm_variational:
                [optim.step() for optim in self.optimizers]
            else:
                self.optimizers[0].step()

            epoch_loss += loss.item()
            epoch_ll += log_likelihood.item()
            epoch_kl += kl_divergence.item()

        return epoch_loss, (-epoch_ll, epoch_kl)