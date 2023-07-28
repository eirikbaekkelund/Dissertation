from collections.abc import Iterable
from abc import abstractmethod
import torch
import gpytorch
import numpy as np
import sys
from alfi.models import VariationalLFM
from alfi.utilities.torch import is_cuda
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torch.nn import Module
from time import time
from pathlib import Path
from typing import List


class TorteTrainer:
    """
    An abstract LFM trainer. Subclasses must implement the `single_epoch` function.

    Parameters
    ----------
    model: The Model.
    optimizers: list of `torch.optim.Optimizer`s. For when natural gradients are used for variational models.
    dataset: Dataset where t_observed (D, T), m_observed (J, T).
    batch_size:
    valid_size: int or float for number of training points or proportion of training points respectively
    test_size: int or float for number of or proportion of training points respectively
    track_parameters: the keys into `named_parameters()` of parameters that the trainer should track. The
                      tracked parameters can be accessed from `parameter_trace`
    train_mask: boolean mask
    """
    def __init__(self,
                 model: Module,
                 optimizers: List[torch.optim.Optimizer],
                 dataset,
                 batch_size=1,
                 valid_size=0.,
                 test_size=0.,
                 device_id=None,
                 track_parameters=None,
                 train_mask=None,
                 checkpoint_dir=None):
        self.model = model
        self.num_epochs = 0
        self.optimizers = optimizers
        self.dataset = dataset
        self.use_natural_gradient = len(self.optimizers) > 1
        self.batch_size = batch_size

        # Dataset splits
        dataset_size = len(dataset)
        if isinstance(valid_size, int):
            valid_size /= dataset_size
        if isinstance(test_size, int):
            test_size /= test_size
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        valid_split = int(np.floor(valid_size * dataset_size))
        test_split = int(np.floor(test_size * dataset_size))
        self.valid_indices = indices[:valid_split]
        self.test_indices = indices[valid_split:test_split + valid_split]
        self.train_indices = indices[test_split + valid_split:]
        self.data_loader = self.test_loader = self.valid_loader = None
        self.set_loaders()

        self.losses = None
        self.train_mask = train_mask
        self.checkpoint_dir = checkpoint_dir
        self.parameter_trace = None
        self.device_id = device_id

        if track_parameters is not None:
            named_params = dict(model.named_parameters())
            self.parameter_trace = {key: [named_params[key].detach()] for key in track_parameters}

    def set_loaders(self):
        valid_data = Subset(self.dataset, self.valid_indices)
        test_data = Subset(self.dataset, self.test_indices)
        train_data = Subset(self.dataset, self.train_indices)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        self.valid_loader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False)
        self.data_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)

    def train(self, epochs=20, report_interval=1, reporter_callback=None, **kwargs):
        """
        Parameters:
            epochs: the number of training epochs to run
            report_interval: every report_interval epochs, a progress string will be printed
            reporter_callback: function called every report_interval
        """
        self.model.train()

        losses = list()
        times = list()
        end_epoch = self.num_epochs+epochs

        for epoch in range(epochs):
            epoch_loss = self.single_epoch(epoch=self.num_epochs, **kwargs)
            t = time()
            times.append((t, epoch_loss))

            if isinstance(epoch_loss, Iterable):
                epoch_loss, loss_breakdown = epoch_loss
            else:
                loss_breakdown = [epoch_loss]

            if (epoch % report_interval) == 0:
                if reporter_callback is not None:
                    reporter_callback(self.num_epochs)
                print('Epoch %03d/%03d - Loss: %.2f ' % (
                    self.num_epochs + 1, end_epoch, epoch_loss), end='')
                if len(loss_breakdown) > 1:
                    print(' '.join(map(lambda l: '%.2f' % l, loss_breakdown[1:])), end='')

                self.print_extra()
                if self.checkpoint_dir is not None:
                    self.model.save(self.checkpoint_dir / f'epoch{epoch}')
            losses.append(loss_breakdown)

            self.after_epoch()
            self.num_epochs += 1

        losses = torch.tensor(losses).cpu().numpy()
        if self.losses is None:
            self.losses = np.empty((0, losses.shape[1]))
        self.losses = np.concatenate([self.losses, losses], axis=0)
        return times

    @abstractmethod
    def single_epoch(self, epoch=0, **kwargs):
        raise NotImplementedError

    def set_optimizers(self, optimizers):
        self.optimizers = optimizers

    def print_extra(self):
        print('')

    def after_epoch(self):
        if self.parameter_trace is not None:
            params = dict(self.model.named_parameters())
            for key in params:
                if key in self.parameter_trace:
                    self.parameter_trace[key].append(params[key].detach().clone())

    def save(self, folder_prefix='model', save_model=True, save_losses=True, save_optimizer=False, additional=None):
        """
        Save the model and trainer information.

        :param folder_prefix: the folder name containing the saved files will be folder_prefix-timestamp
        :param save_model:
        :param save_losses:
        :param save_optimizer:
        :param additional: object or list of objects to save alongside the default items.
        :return:
        """
        timestamp = int(time())
        path = Path(f'./{folder_prefix}-{timestamp}/')
        if path.exists():
            raise IOError('Path seems to already exist.')
        path.mkdir()

        torch.save((self.train_indices, self.test_indices, self.valid_indices), path / 'indices.pt')
        if save_model:
            torch.save(self.model.state_dict(), path / 'model.pt')

        if save_losses:
            torch.save(self.losses, path / 'losses.pt')

        if save_optimizer:
            for i, optimizer in enumerate(self.optimizers):
                torch.save(optimizer.state_dict(), path / 'optim[{i}].pt')

        if additional is not None:
            torch.save(additional, path / 'additional.pt')
        return timestamp

    def load(self, file_prefix='model', timestamp=''):
        """
        Loads optionally the model state dict, losses array, optimizer state dict, in that order.

        :param file_prefix:
        :param timestamp:
        """
        path = Path(f'./{file_prefix}-{timestamp}/')
        print('Loading from', path)
        if (path / 'indices.pt').exists():
            indices = torch.load(path / 'indices.pt')
            self.train_indices, self.test_indices, self.valid_indices = indices
            self.set_loaders()
        if (path / 'model.pt').exists():
            state_dict = torch.load(path / 'model.pt', map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
        if (path / 'losses.pt').exists():
            self.losses = torch.load(path / 'losses.pt')
        for i, optim in enumerate([f'optim[{i}].pt' for i in range(len(self.optimizers))]):
            if (path / optim).exists():
                self.optimizers[i].load_state_dict(torch.load(path / optim))

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
        
            data_input = data_input.squeeze(0).float()
        
            output = self.model(data_input, step_size=self.step_size)
            y_target = y.t().squeeze(-1)
            
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