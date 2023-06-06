import torch
from models import ExactGPModel
import data_loader as dl

def train_gp(x, y, n_hours, n_iter, lr, optim, likelihood, mean_module, covar_module):
    """
    Train the GP model

    Args:
        train_x (torch.Tensor): training data
        train_y (torch.Tensor): training labels
        n_iter (int): number of iterations
        lr (float): learning rate
        optim (str): optimizer
        mean_module (gpytorch.means.Mean): mean module
        covar_module (gpytorch.kernels.Kernel): covariance module

    Returns:
        ExactGPModel: trained GP model
    """

    
    y_copy = torch.zeros(y.shape)
    y_copy[:, :, 0] = y[:, :, 0]
    
    for k in range(y.shape[1]):
        time, y_sample = dl.extract_time_series(time=x, y=y, idx=k)
        time_train, y_train, time_test, _ = dl.train_test_split_1d(time, y_sample, n_hours=n_hours)
        model = ExactGPModel(time_train, y_train, likelihood, mean_module, covar_module)
        model._train(n_iter, lr, optim)
        preds_test, _ = model.predict(time_test)
        
        y_copy[len(time_train):, k, 0] = preds_test.mean
        
    return y_copy

def create_gp_covars(time, pv_series, index_to_drop, likelihood, mean_module, covar_module):

    # Create the complementary indices to keep
    indices_to_keep = [i for i in range(pv_series.size(1)) if i != index_to_drop]

    # Select the desired time series
    pv_input = torch.index_select(pv_series, dim=1, index=torch.tensor(indices_to_keep))
    pv_pred = train_gp(x=time,
            y=pv_input,
            n_hours=2,
            n_iter=500,
            lr=0.05,
            optim='Adam',
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module)

    covars = torch.cat((time, pv_pred.squeeze()), dim=1)
    target = pv_series[:, index_to_drop, :]

    return covars, target
