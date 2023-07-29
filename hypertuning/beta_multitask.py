import torch
import optuna
from hypertuning.base import GPQuasiPeriodic
from models import MultitaskGPModel

# TODO maybe not inherit from GPQuasiPeriodic

class MultitaskBetaQPGP(GPQuasiPeriodic):
    """
    Multitask Quasi-Periodic GP hyperparameter optimization.
    Beta likelihood with mean parametrization
    """
    def instantiate_model(self):
        """ 
        Create the model.
        """
        self.model = MultitaskGPModel(**self.inputs)
        self.model.fit(n_iter=200, lr=0.2, verbose=False)
    
    def objective(self, trial: optuna.trial.Trial):
        """ 
        Objective function for the hyperparameter optimization.
        """
        self.num_latents = trial.suggest_int('num_latents', 2, self.num_tasks)
        self.sample_params(trial)
        self.inputs['num_latents'] = self.num_latents
        
        n_skips = 0

        # TODO kernels could change once weather data is included

        for (x_tr, y_tr), (x_te, y_te) in zip(self.train_loader, self.test_loader):
            
            self.inputs['X'] = x_tr
            self.inputs['y'] = y_tr

            if y_tr.isnan().any() or y_te.isnan().any():
                # remove the second dimension of the tensors
                # if either y_tr or y_te has a nan in it
                nan_tr = y_tr.isnan().any(dim=1)
                nan_te = y_te.isnan().any(dim=1)

                x_tr = x_tr[~nan_tr]
                y_tr = y_tr[~nan_tr]
                x_te = x_te[~nan_te]
                y_te = y_te[~nan_te]

            if y_tr.size(0) == 0 or y_te.size(0) == 0:
                continue
            try:
                self.instantiate_model()
            except: # NotPSDError
                n_skips += 1
                print(f'Not PSD error, skip nr.: {n_skips}')
                continue

            pred_dist = self.model.predict(x_te)
            metric = self.metric(pred_dist, y_te)
            self.metrics.append(metric)
        
        return torch.tensor(self.metrics).mean().item()
