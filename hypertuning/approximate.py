import torch
import numpy as np
import optuna
from hypertuning.base import GPQuasiPeriodic
from models.approximate import ApproximateGPBaseModel


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
            'mean_init_std' : 1e-4,
        }
        self.inputs['config'] = self.config
    
    def instantiate_model(self):
        """ 
        Create the model.
        """
        self.model = ApproximateGPBaseModel(**self.inputs)
        self.model.fit(**self.train_config, verbose=False)
    
    def store_metrics(self, metrics : list, percentiles_coverage : list):
        self.metrics = metrics
        self.percentiles_coverage = percentiles_coverage
    
    def get_metrics(self):
        assert self.metrics is not None, 'Metrics not yet computed.'
        assert self.percentiles_coverage is not None, 'Percentiles coverage not yet computed.'
        
        return self.metrics, self.percentiles_coverage

    def objective(self, trial : optuna.trial.Trial):
        """ 
        Objective function for the hyperparameter optimization.
        """
        self.sample_params(trial)
        
        n_skips = 0
        metrics = []
        percentiles_coverage = []
        
        for (x_tr, y_tr), (x_te, y_te) in zip(self.train_loader, self.test_loader):
                
            self.inputs['X'] = x_tr
            
            for i in range(y_tr.shape[1]):
                if y_tr[:,i].isnan().any() or y_te[:,i].isnan().any():
                    
                    n_skips += 1
                    print(f'NaN error, skip nr.: {n_skips}')
                    continue
                
                self.config['num_inducing_points'] = x_tr.size(0)
                self.inputs['config'] = self.config
                self.inputs['y'] = y_tr[:,i]
                try:
                    self.instantiate_model()
                except:
                    n_skips += 1
                    print(f'NotPSDError, skip nr.: {n_skips}')
                    continue
                
                pred_dist = self.model.predict(x_te)
                metric = self.metric(pred_dist, y_te[:,i])
                metrics.append(metric)
                
                preds = pred_dist.sample((50,))
                lower, upper = np.percentile(preds, [2.5, 97.5], axis=0)
                lower, upper = lower.mean(axis=0), upper.mean(axis=0)
                
                # find the percentage of test points that are within the 95% confidence interval
                y = y_te[:,i].detach().numpy()
               
                pct_inside = 100 * ((y > lower) & (y < upper)).sum() / y.shape[0]
                percentiles_coverage.append(pct_inside)
                
                print(f'Percentiles coverage: {pct_inside:.2f}%')
                print(f'NLPD: {metric:.4f}')
        
        self.store_metrics(metrics, percentiles_coverage)
        return torch.tensor(metrics).mean().item() + n_skips