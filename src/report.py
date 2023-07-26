import torch
import gpytorch
import numpy as np
from metric_report import IndepSVIGPReport, MultitaskGPReport
import data_loader as dl
from data_loader import PVDataLoader, PVDataGenerator
from metric_report import IndepSVIGPReport, MultitaskGPReport
from kernel import generate_quasi_periodic
from beta_likelihood import BetaLikelihood_MeanParametrization

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':

    # data parameters
    DAY_INIT = 0
    DAY_MIN = 8
    DAY_MAX = 16
    N_DAYS = 300
    N_DAYS_TRAIN = 5
    MINUTE_INTERVAL = 5
    DAILY_DATA_POINTS = (DAY_MAX - DAY_MIN) * 60 / MINUTE_INTERVAL
    N_HOURS_PRED = 2
    N_SYSTEMS = 55
    RADIUS = 0.35
    CIRCLE_COORDS = (55, -1.5)

    generator = PVDataGenerator(n_days=N_DAYS,
                    day_init=DAY_INIT,
                    n_systems=N_SYSTEMS,
                    coords=CIRCLE_COORDS,
                    minute_interval=MINUTE_INTERVAL,
                    day_min=DAY_MIN,
                    day_max=DAY_MAX,
                    folder_name='pv_data',
                    file_name_pv='pv_data_clean.csv',
                    file_name_location='location_data_clean.csv',
                    distance_method='circle',
                    drop_nan=False)

    X, y = generator.get_time_series()

    N_TASKS = y.size(-1)
    NUM_LATENT = 5
    
    x_list, y_list = dl.cross_val_fold(X, y, N_DAYS_TRAIN, DAILY_DATA_POINTS)
    x_train, y_train, x_test, y_test =  dl.train_test_split_fold(x_list, y_list, N_HOURS_PRED, DAILY_DATA_POINTS, DAY_MIN, DAY_MAX)
    
    train_loader = PVDataLoader(x_train, y_train)
    test_loader = PVDataLoader(x_test, y_test)

    # covar_module = generate_quasi_periodic(1)
    # mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([1]))
    # likelihood = BetaLikelihood_MeanParametrization(scale=1, batch_shape=torch.Size([1]))

    # svi_beta = IndepSVIGPReport(train_loader=train_loader,
    #                             test_loader=test_loader,
    #                             verbose=True,
    #                             mean_module=mean_module,
    #                             covar_module=covar_module,
    #                             likelihood=likelihood)
    # svi_beta.generate_metrics()
    # svi_beta.write_report('svi_beta')

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=N_TASKS)
    mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([NUM_LATENT]))
    covar_module = generate_quasi_periodic(num_latents=NUM_LATENT)

    multitask_gp = MultitaskGPReport(train_loader=train_loader,
                                    test_loader=test_loader,
                                    verbose=True,
                                    mean_module=mean_module,
                                    covar_module=covar_module,
                                    likelihood=likelihood,
                                    num_latents=NUM_LATENT)
    multitask_gp.generate_metrics()
    multitask_gp.write_report('svi_multitask_gp')


