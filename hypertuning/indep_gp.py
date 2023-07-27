import torch
import gpytorch
import optuna
from hypertuning.base import GPQuasiPeriodic
from likelihoods.beta import BetaLikelihood_MeanParametrization
from models.approximate_gp import ApproximateGPBaseModel

class HyperOptBetaGP(GPQuasiPeriodic):
    def __init__(self,
                train_loader : torch.utils.data.DataLoader,
                test_loader : torch.utils.data.DataLoader,
                ):
        super().__init__(train_loader=train_loader,
                         test_loader=test_loader,
                         num_latents=1)
        self.config = {'type' : 'stochastic',
                        'name' : 'mean_field',
                        'mean_init_std' : 1,
                        }

    def get_likelihood(self, trial : optuna.trial.Trial):
        """ 
        Sample the likelihood of the model.
        """
        likelihood_scale = trial.suggest_int('likelihood_scale', 1, 60, step=5)        
        likelihood = BetaLikelihood_MeanParametrization(scale=likelihood_scale)
                                                        
        return likelihood         

    def instantiate_model(self, 
                          x_tr : torch.Tensor, 
                          y_tr : torch.Tensor, 
                          jitter : float):
        """ 
        Create a model instance.
        """
      
        self.config['num_inducing_points'] = x_tr.size(0)
        self.inputs['x_train'] =  x_tr
        self.inputs['y_train'] =  y_tr
        self.inputs['jitter'] = jitter
        self.inputs['config'] = self.config
        
        model = ApproximateGPBaseModel(**self.inputs)
        model.fit(n_iter=10, lr=0.2, verbose=True)
        return model

    def metric(self, y_dist, target):
        """
        Metric to optimize.
        """
        return gpytorch.metrics.negative_log_predictive_density(y_dist, target).median()
    
    def objective(self, trial : optuna.trial.Trial):
        """
        Objective function to minimize.
        """
        losses = []
        mean_module, kernel = self.sample_params(trial)
        likelihood = self.get_likelihood(trial)
       
        self.inputs = {'likelihood': likelihood,
                        'mean_module': mean_module,
                        'covar_module': kernel,
                        }
        # iterate over folds 
       
        for (x_tr, y_tr), (x_te, y_te) in zip(self.train_loader, self.test_loader):
            # iterate over each series in the fold
            for i in range(y_tr.shape[1]):
                jitter = 1e-4
                # fit model for each series
                try:
                    model = self.instantiate_model(x_tr, y_tr[:,i], jitter, trial)
                
                except:
                    print('Not PSD error, adding jitter')
                    jitter *= 10
                    try:
                        model = self.instantiate_model(x_tr, y_tr[:,i], jitter, trial)
                    except:
                        continue

                # get predictive distribution
                with torch.no_grad():
                    trained_pred_dist = model.likelihood(model(x_te))
                
                # calculate metric
                nlpd = self.metric(trained_pred_dist, y_te[:,i])
                losses.append(nlpd)
        
        return torch.mean(torch.tensor(losses), dtype=torch.float32)

