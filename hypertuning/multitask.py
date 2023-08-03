import torch
import optuna
from hypertuning.base import GPQuasiPeriodic
from models import MultitaskGPModel
from gpytorch.utils.errors import NotPSDError

class MultitaskBetaQPGP(GPQuasiPeriodic):
    """
    Multitask Quasi-Periodic GP hyperparameter optimization.
    Beta likelihood with mean parametrization
    """
    def __init__(self, 
                 train_loader: torch.utils.data.DataLoader, 
                 test_loader: torch.utils.data.DataLoader):
        
        super().__init__(train_loader, test_loader)
        
        _, y = next(iter(train_loader))
        self.num_tasks = y.size(-1)
    
    def instantiate_model(self):
        """ 
        Create the model.
        """
        # TODO set learning configuration
        self.model = MultitaskGPModel(**self.inputs)
        self.model.fit(**self.train_config, verbose=False)
    
    def objective(self, trial: optuna.trial.Trial):
        """ 
        Objective function for the hyperparameter optimization.
        """
        self.num_latents = trial.suggest_int('num_latents', 2, self.num_tasks)
        self.sample_params(trial)
        self.inputs['num_latents'] = self.num_latents
        
        n_skips = 0
        nlpds = []

        for (x_tr, y_tr), (x_te, y_te) in zip(self.train_loader, self.test_loader):
            
            # remove any nan values from the data 
            # worth noting that there is <5% nan vals for each system
            # so shouldn't have a massive impact on the results

            if y_tr.isnan().any() or y_te.isnan().any():
                # remove the second dimension of the tensors
                # if either y_tr or y_te has a nan in it
                nan_tr = y_tr.isnan().any(dim=1)
                nan_te = y_te.isnan().any(dim=1)

                x_tr = x_tr[~nan_tr]
                y_tr = y_tr[~nan_tr]
                x_te = x_te[~nan_te]
                y_te = y_te[~nan_te]

            # multitask requires a 2D tensor with n_tasks > 1
            # to be useful
            
            if y_tr.size(0) <= 1 or y_te.size(0) <= 1:
                n_skips += 1
                print(f'Empty tensor error, skip nr.: {n_skips}')
                continue

            self.inputs['X'] = x_tr
            self.inputs['y'] = y_tr
            try:
                self.instantiate_model()
            except NotPSDError:
                try:
                    self.inputs['jitter'] = 1e-2
                    self.instantiate_model()
                except:
                    continue

            dist = self.model.predict(x_te, pred_type='dist')
            
            samples = dist.sample(sample_shape=torch.Size([30]))
            _lower, _upper = self.model.confidence_region(samples)
            
            for i in range(y_te.shape[1]):
                lower, upper = _lower[:, i], _upper[:, i]
                y = y_te[:, i].numpy()
                inside = ((y >= lower) & (y <= upper)).sum()
                print(f'Percentage inside 95% CI: { ( inside / y_te.size(0) ) * 100:.2f}%')
            
            neg_log_pred_dens = self.metric(dist, y_te)
            nlpds.append(neg_log_pred_dens)
        
        return torch.tensor(nlpds).mean().item()