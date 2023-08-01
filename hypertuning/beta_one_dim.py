import torch
import optuna
from hypertuning.base import GPQuasiPeriodic
from models.approximate import ApproximateGPBaseModel
from alive_progress import alive_bar
from time import sleep

class BetaQPGPOneDim(GPQuasiPeriodic):
    """ 
    One dimensional Quasi-Periodic GP hyperparameter optimization.
    Beta likelihood with mean parametrization
    """
    def __init__(self, 
                 train_loader: torch.utils.data.DataLoader, 
                 test_loader: torch.utils.data.DataLoader,
                 ):
        super().__init__(train_loader, test_loader)
        self.config = {
            'type' : 'stochastic',
            'name' : 'mean_field',
            'mean_init_std' : 1,
        }
        self.inputs = {'config' : self.config,
                       'jitter' : 1e-4,
                       }
        self.metrics = []
    
    def instantiate_model(self):
        """ 
        Create the model.
        """
        self.model = ApproximateGPBaseModel(**self.inputs)
        self.model.fit(**self.train_config)

    def objective(self, trial : optuna.trial.Trial):
        """ 
        Objective function for the hyperparameter optimization.
        """
        self.sample_params(trial)
        
        n_skips = 0
       
        total = len(self.train_loader)
        
        with alive_bar(total) as bar:
            for (x_tr, y_tr), (x_te, y_te) in zip(self.train_loader, self.test_loader):
                
                self.inputs['X'] = x_tr
                
                for i in range(y_tr.shape[1]):
                    if y_tr[:,i].isnan().any() or y_te[:,i].isnan().any():
                        sleep(0.01)
                        bar()
                        n_skips += 1
                        print(f'NaN error, skip nr.: {n_skips}')
                        continue
                    
                    self.config['num_inducing_points'] = x_tr.size(0)
                    self.inputs['config'] = self.config
                    self.inputs['y'] = y_tr[:,i]
                    try:                
                        self.instantiate_model()
                    except: # NotPSDError    
                        n_skips += 1
                        print(f'Not PSD error, skip nr.: {n_skips}')
                        continue
                    
                    pred_dist = self.model.predict(x_te)
                    metric = self.metric(pred_dist, y_te[:,i])
                    self.metrics.append(metric)

                    sleep(0.01)
                    bar()
                        
        return torch.tensor(self.metrics).mean().item()