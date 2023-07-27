from abc import ABC, abstractmethod
import gpytorch
import torch
import numpy as np
import pandas as pd
from gpytorch.metrics import negative_log_predictive_density as nlpd
from models.approximate import ApproximateGPBaseModel
from models.multitask import MultitaskGPModel
from data_loader import PVDataLoader

def mae(y_pred, y_true):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
        
    return np.mean(np.abs(y_pred - y_true))

class MetricReport(ABC):
    """ 
    Base class for metric reporters

    """
    def __init__(self, train_loader : PVDataLoader, test_loader : PVDataLoader, verbose : bool = False):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.verbose = verbose
    
    @abstractmethod
    def metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate_metrics(self):
        pass

    @abstractmethod
    def write_report(self, *args, **kwargs):
        pass

class IndepSVIGPReport(MetricReport):
    """ 
    
    """
    def __init__(self,
                train_loader : PVDataLoader,
                test_loader : PVDataLoader,
                mean_module : gpytorch.means.Mean,
                covar_module : gpytorch.kernels.Kernel,
                likelihood : gpytorch.likelihoods.Likelihood,
                verbose : bool = False):
        super().__init__(train_loader, test_loader, verbose)
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood
        
        self.config = {
            'type' : 'stochastic',
            'name' : 'mean_field',
            'mean_init_std' : 1,
        }
        self.inputs = {
                    'likelihood': likelihood,
                    'mean_module': mean_module,
                    'covar_module': covar_module,
                    }

    def fit_model(self, x_tr, y_tr, jitter):
        """ 
        Instantiate and fit a Beta GP
        """
        
        self.config['num_inducing_points'] = x_tr.size(0)
        self.inputs['X'] = x_tr
        self.inputs['y'] = y_tr
        self.inputs['config'] = self.config
        self.inputs['jitter'] = jitter

        model = ApproximateGPBaseModel(**self.inputs)
        model.fit(n_iter=150, lr=0.2, verbose=self.verbose)
    
        return model
    
    def metrics(self, pred_dist, y_te, model):
        neg_log_pred = nlpd(pred_dist, y_te).median().item()
        pred_sample = pred_dist.sample((50,))
        
        mean = pred_sample.mean(dim=0)
        median = pred_sample.median(dim=0).values
        mode = model.likelihood.mode()

        mae_mean = mae(mean, y_te)
        mae_median = mae(median, y_te)
        mae_mode = mae(mode, y_te)

        return neg_log_pred, mae_mean, mae_median, mae_mode

    def generate_metrics(self, ):
        """ 
        Calculates the negative log predictive density metric

        Args:
            report_name (str): the name of the report (metric name and model name)
        Returns:
            nlpd (torch.Tensor): the negative log predictive density
        """
        self.nlpd_list = []
        self.mae_mean_list = []
        self.mae_median_list = []
        self.mae_mode_list = []

        # iterate over folds 
        for (x_tr, y_tr), (x_te, y_te) in zip(self.train_loader, self.test_loader):
            # iterate over each series in the fold
            for i in range(y_tr.shape[1]):
                jitter = 1e-4
                if y_tr[:,i].isnan().any() or y_te[:,i].isnan().any():
                    continue
                # fit model for each series
                try:
                    model = self.fit_model(x_tr, y_tr[:,i], jitter)
                
                except:
                    print('Not PSD error, adding jitter')
                    jitter = 1e-2
                    try:
                        model = self.fit_model(x_tr, y_tr[:,i], jitter)
                    except:
                        print('Not PSD error, skipping series')
                        continue

                # get predictive distribution
                with torch.no_grad():
                    trained_pred_dist = model.likelihood(model(x_te))
                
                # calculate metrics
                nlpd, mae_mean, mae_median, mae_mode = self.metrics(trained_pred_dist, y_te[:,i], model)
                self.nlpd_list.append(nlpd)
                self.mae_mean_list.append(mae_mean)
                self.mae_median_list.append(mae_median)
                self.mae_mode_list.append(mae_mode)
    
    def write_report(self, report_name):
        
        """ 
        Generates a report of the metrics
        """
        report = pd.DataFrame({'nlpd': self.nlpd_list,
                            'mae_mean': self.mae_mean_list,
                            'mae_median': self.mae_median_list,
                            'mae_mode': self.mae_mode_list})
        report.to_csv(f'{report_name}.csv')

class MultitaskGPReport(MetricReport):
    """ 
    Negative log predictive density metric reporter   
    """
    def __init__(self,
                train_loader : PVDataLoader,
                test_loader : PVDataLoader,
                mean_module : gpytorch.means.Mean,
                covar_module : gpytorch.kernels.Kernel,
                likelihood : gpytorch.likelihoods.Likelihood,
                num_latents : int,
                verbose : bool = False):
        super().__init__(train_loader, test_loader, verbose)
        _, y = self.train_loader.__getitem__(0)
        self.n_tasks = y.size(-1)
        self.inputs = {
                'mean_module': mean_module,
                'covar_module': covar_module,
                'likelihood': likelihood,
                'num_latents' : num_latents,
                }        

    def fit_model(self, x_tr, y_tr, jitter):
        self.inputs['x_train'] = x_tr
        self.inputs['y_train'] = y_tr
        self.inputs['jitter'] = jitter

        model = MultitaskGPModel(**self.inputs)
        model.fit(n_iter=300, lr=0.2, verbose=self.verbose)
        return model
   
    def metrics(self, pred_dist, y_te):
        neg_log_pred = nlpd(pred_dist, y_te).median().item()
        mae_mean = mae(pred_dist.mean, y_te)
        return neg_log_pred, mae_mean

    def generate_metrics(self):
        """ 
        Calculates and stores the metrics
        """

        self.nlpd_list = []
        self.mae_mean_list = []
        # TODO add more metrics if we get Beta Multitask GP working

        # iterate over folds 
        for (x_tr, y_tr), (x_te, y_te) in zip(self.train_loader, self.test_loader):
            # iterate over each series in the fold
            jitter = 1e-4
            # fit model for each series
            
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
                model = self.fit_model(x_tr, y_tr, jitter)
            except:
                print('Not PSD error, adding jitter')
                jitter *= 10
                model = self.fit_model(x_tr, y_tr, jitter)

            # get predictive distribution
            with torch.no_grad():
                pred_dist = model.likelihood(model(x_te))
            
            # calculate metric
            nlpd, mae_mean = self.metrics(pred_dist, y_te)
            self.nlpd_list.append(nlpd)
            self.mae_mean_list.append(mae_mean)
          
    
    def write_report(self, report_name):
        """ 
        Generates a report of the metrics

        Args:
            report_name (str): the name of the report (metric name and model name)
        Returns:
            nlpd (torch.Tensor): the negative log predictive density
        """

        report = pd.DataFrame({'nlpd': self.nlpd_list,
                            'mae_mean': self.mae_mean_list})
        report.to_csv(f'{report_name}.csv')