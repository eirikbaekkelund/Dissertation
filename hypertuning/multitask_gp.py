import torch
import gpytorch
import optuna
import numpy as np
from hypertuning.base import GPQuasiPeriodic
from likelihoods.beta import MultitaskBetaLikelihood
from models.multitask_gp import MultitaskGPModel


# TODO maybe not inherit from GPQuasiPeriodic

class HyperOptMultitaskGP(GPQuasiPeriodic):
    def __init__(self,
            train_loader : torch.utils.data.DataLoader,
            test_loader : torch.utils.data.DataLoader,
            num_latents : int = 5
            ):
        super().__init__(train_loader=train_loader,
                         test_loader=test_loader,
                         num_latents=num_latents)

        self.inputs =  {'num_latents': num_latents}
        
    # TODO make beta likelihood for SVI multitask

    def get_likelihood(self, trial : optuna.trial.Trial):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_tasks)
        return likelihood
                                        

    def instantiate_model(self, 
                          x_tr : torch.Tensor, 
                          y_tr : torch.Tensor, 
                          jitter : float,
                        ):
        """ 
        Create a model instance.
        """
        self.inputs['x_train'] =  x_tr
        self.inputs['y_train'] =  y_tr
        self.inputs['jitter'] = jitter     

        model = MultitaskGPModel(**self.inputs)
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
        likelihood = self.get_likelihood(trial)
        mean_module, kernel = self.sample_params(trial)
        
        self.inputs['likelihood'] = likelihood
        self.inputs['mean_module'] = mean_module
        self.inputs['covar_module'] = kernel

        # iterate over folds 
        for (x_tr, y_tr), (x_te, y_te) in zip(self.train_loader, self.test_loader):
            # iterate over each series in the fold
            self.n_tasks = y_tr.shape[1]
            for i in range(y_tr.shape[1]):
                jitter = 1e-4
                # fit model for each series
                try:
                    model = self.instantiate_model(x_tr, y_tr, jitter, trial)
                
                except:
                    print('Not PSD error, adding jitter')
                    jitter *= 10
                    model = self.instantiate_model(x_tr, y_tr, jitter, trial)

                # get predictive distribution
                with torch.no_grad():
                    trained_pred_dist = model.likelihood(model(x_te))
                
                # calculate metric
                nlpd = self.metric(trained_pred_dist, y_te)
                losses.append(nlpd)
        
        return torch.mean(torch.tensor(losses), dtype=torch.float32)