import torch
import numpy as np
import pandas as pd
from models.baselines import (YesterdayForecast, 
                              HourlyAverage, 
                              Persistence, 
                              fit_var, 
                              fit_exp, 
                              fit_simple_exp,
                              var_exp_simulation)
from metrics import mean_absolute_error, get_mean_ci, inside_ci, nlpd_holt
from models import ApproximateGPBaseModel, MultitaskGPModel
from gpytorch.metrics import negative_log_predictive_density as NLPD
from data import PVDataGenerator, PVDataLoader
from data.utils import train_test_split_fold, cross_val_fold, check_model_inputs
from kernels import get_mean_covar
from likelihoods import (BetaLikelihood_MeanParametrization,
                         MultitaskBetaLikelihood)
from pv_plot import plot_forecast_mae, boxplot_models, boxplot_forecast_horizon


# seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# UGLY CODE FOR RUNNING EXPERIMENTS, NO TIME TO CLEAN
# TODO clean code to make it more pythonic

if __name__ == '__main__':
    # data parameters
    DAY_INIT = 0
    DAY_MIN = 8
    DAY_MAX = 16
    # 90 days is roughly a season
    N_DAYS = 90
    N_DAYS_TRAIN = 5
    MINUTE_INTERVAL = 5
    DAILY_POINTS = (DAY_MAX - DAY_MIN) * 60 // MINUTE_INTERVAL
    TOTAL_POINTS = int(N_DAYS * DAILY_POINTS)
    INTERVAL = int(DAILY_POINTS * N_DAYS_TRAIN)
    N_HOURS_PRED = 2
    PRED_POINTS = N_HOURS_PRED * 60 // MINUTE_INTERVAL
    N_SYSTEMS = 15
    RADIUS = 0.35
    CIRCLE_COORDS = (55, -1.5)
    plot_results = True

    models_list =['persistence', 'yesterday', 'hourly_avg', 'var', 
                  'simple_exp', 'exp', 'gp', 'multitask_gp']
    seasons = ['winter', 'spring', 'summer', 'fall']
    # gp model configuration
    jitter = 1e-3
    config = {  'type': 'stochastic',
                'name': 'mean_field',
                }
    inputs ={
        'jitter': jitter,
        'likelihood': BetaLikelihood_MeanParametrization(scale=25),
        'learn_inducing_locations': False
    }
    num_latent = 8 # TODO find from hyperparameter tuning


    # TODO add seasons as argument and save results in a folder with the name of the season
    for season in seasons:
        loader = PVDataGenerator(n_days=N_DAYS,
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
                        season=season,
                        drop_nan=False)

        X, Y = loader.get_time_series()
        x_list, y_list = cross_val_fold(X, Y, N_DAYS_TRAIN, DAILY_POINTS)
        x_train, y_train, x_test, y_test =  train_test_split_fold(x_list, y_list, N_HOURS_PRED, MINUTE_INTERVAL, DAY_MIN, DAY_MAX)

        train_loader = PVDataLoader(x_train, y_train)
        test_loader = PVDataLoader(x_test, y_test)
        
        var_j = 0
        exp_j = 0
        simple_exp_j = 0
        yesterday_j = 0
        hourly_avg_j = 0
        persist_j = 0
        gp_j = 0
        multitask_j = 0

        df_var = None
        df_exp = None
        df_simple_exp = None
        df_yesterday = None
        df_hourly_avg = None
        df_persistence = None
        df_gp = None
        df_multitask_gp = None
        df_multitask_gp_nlpd = None
        psd_error = False

        for (x_train, y_train), (x_test, y_test) in zip(train_loader, test_loader):
            
            x_train, y_train, x_test, y_test = check_model_inputs(x_train, y_train, x_test, y_test)
            if x_train is None:
                continue

            elif y_train.shape[-1] < num_latent:
                continue
            
            for model in models_list:
                if model not in ['gp', 'multitask_gp', 'holt'] and isinstance(y_train, torch.Tensor):
                    y_train = y_train.numpy()
                    y_test = y_test.numpy()
                
                elif model in ['gp', 'multitask_gp', 'holt'] and not isinstance(y_train, torch.Tensor):
                    y_train = torch.from_numpy(y_train)
                    y_test = torch.from_numpy(y_test)

                if model == 'var':
                    
                    y_pred = fit_var(y_train, PRED_POINTS)
                    mae = mean_absolute_error(y_test, y_pred)
                    for k in range(mae.shape[-1]):
                        if df_var is None:
                            df_var = pd.DataFrame({f'var_mae{var_j}': mae[:,k]})
                        else:
                            new_df = pd.DataFrame({f'var_mae{var_j}': mae[:,k]})
                            df_var = pd.concat([df_var, new_df], axis=1)
                        var_j += 1
                if model == 'exp':
                
                    for k in range(y_train.shape[-1]): 
                        y_pred, fitted_model = fit_exp(y_train[:,k], PRED_POINTS)
                        var = var_exp_simulation(fitted_model, y_pred, n_pred=PRED_POINTS)
                    
                        if np.isnan(y_pred).any():
                            continue
                        mae = mean_absolute_error(y_test[:,k], y_pred)
                        nlpd = nlpd_holt(y_pred, var, y_test[:,k])

                        if df_exp is None:
                            df_exp = pd.DataFrame({f'exp_mae{exp_j}': mae})
                            df_exp_nlpd = pd.DataFrame({f'exp_nlpd{exp_j}': nlpd})
                        else:
                            new_df = pd.DataFrame({f'exp_mae{exp_j}': mae})
                            df_exp = pd.concat([df_exp, new_df], axis=1)

                            new_df = pd.DataFrame({f'exp_nlpd{exp_j}': nlpd})
                            df_exp_nlpd = pd.concat([df_exp_nlpd, new_df], axis=1)
                        
                        exp_j += 1
                
                elif model == 'simple_exp':
                    for k in range(y_train.shape[-1]): 
                        y_pred, fitted_model = fit_simple_exp(y_train[:,k], PRED_POINTS)
                        var = var_exp_simulation(fitted_model, y_pred, n_pred=PRED_POINTS)
                    
                        if np.isnan(y_pred).any():
                            continue
                        mae = mean_absolute_error(y_test[:,k], y_pred)
                        nlpd = nlpd_holt(y_pred, var, y_test[:,k])
                    
                        if df_simple_exp is None:
                            df_simple_exp = pd.DataFrame({f'simple_exp_mae{simple_exp_j}': mae})
                            df_simple_exp_nlpd = pd.DataFrame({f'simple_exp_nlpd{simple_exp_j}': nlpd})
                        else:
                            new_df = pd.DataFrame({f'simple_exp_mae{simple_exp_j}': mae})
                            df_simple_exp = pd.concat([df_simple_exp, new_df], axis=1)

                            new_df = pd.DataFrame({f'simple_exp_nlpd{simple_exp_j}': nlpd})
                            df_simple_exp_nlpd = pd.concat([df_simple_exp_nlpd, new_df], axis=1)
                        simple_exp_j += 1
                    
                elif model == 'yesterday':
                    model = YesterdayForecast(DAY_MIN, DAY_MAX, MINUTE_INTERVAL)
                    y_pred = model.predict(y_train)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    for k in range(mae.shape[-1]):
                        if df_yesterday is None:
                            df_yesterday = pd.DataFrame({f'yesterday_mae{yesterday_j}': mae[:,k]})
                        else:
                            new_df = pd.DataFrame({f'yesterday_mae{yesterday_j}': mae[:,k]})
                            df_yesterday = pd.concat([df_yesterday, new_df], axis=1)
                    
                        yesterday_j += 1
                
                elif model == 'hourly_avg':
                    
                    model = HourlyAverage(DAY_MIN, DAY_MAX, MINUTE_INTERVAL)
                    y_pred = model.predict(y_train)
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    for k in range(mae.shape[-1]):
                        if df_hourly_avg is None:
                            df_hourly_avg = pd.DataFrame({f'hourly_avg_mae{hourly_avg_j}': mae[:,k]})
                        else:
                            new_df = pd.DataFrame({f'hourly_avg_mae{hourly_avg_j}': mae[:,k]})
                            df_hourly_avg = pd.concat([df_hourly_avg, new_df], axis=1)
                        hourly_avg_j += 1
                
                elif model == 'persistence':
                    model = Persistence()
                    y_pred = model.predict(y_train, PRED_POINTS)
                    mae = mean_absolute_error(y_test, y_pred)
                    for k in range(mae.shape[-1]):
                        if df_persistence is None:
                            df_persistence = pd.DataFrame({f'persistence_mae{persist_j}': mae[:,k]})
                        else:
                            new_df = pd.DataFrame({f'persistence_mae{persist_j}': mae[:,k]})
                            df_persistence = pd.concat([df_persistence, new_df], axis=1)
                        persist_j += 1
                
                elif model == 'gp':

                    for k in range(y_train.shape[-1]): 
                        # only for numerical stability (values are already scaled)
                        if torch.isnan(y_test[:,k]).any():
                            continue
                        mean, covar = get_mean_covar()
                        inputs['mean_module'] = mean
                        inputs['covar_module'] = covar
                        # just to avoid floating point errors
                        y = torch.clamp(y_train[:,k], 1e-4, 1 - 1e-4)
                        # remove potential nans
                        if torch.isnan(y).any(): 
                            mask = ~torch.isnan(y)
                            y_masked = y[mask]
                            x_masked = x_train[mask]
                            config['num_inducing_points'] = x_masked.shape[0]
                            inputs['config'] = config
                            inputs['X'] = x_masked
                            inputs['y'] = y_masked
                        else:
                            config['num_inducing_points'] = x_train.shape[0]
                            inputs['config'] = config
                            inputs['X'] = x_train
                            inputs['y'] = y
                        
                        model = ApproximateGPBaseModel(**inputs)
                        # set x back to original if it was masked
                    
                        if k != 0 and (not psd_error or not torch.isnan(y).any()):
                            try:
                                model.warm_start(state_dict)
                            except RuntimeError:
                                print('variational strategy not compatible with warm start. Shape mismatch')
                        try:
                            model.fit(n_iter=200, lr=0.2, use_wandb=True)
                            # save model parameters to pass on next iteration
                            state_dict = model.state_dict()
                            
                            y_pred, lower, upper = model.predict(x_test, pred_type='median')
                            pred_dist = model.predict(x_test, pred_type='dist')
                            
                            pct_inside = inside_ci(lower, upper, y_test[:,k].numpy())
                            mae = mean_absolute_error(y_test[:,k], y_pred)
                            nlpd = NLPD(pred_dist, y_test[:,k]).median(axis=0).values
                    
                            if df_gp is None:
                                df_gp = pd.DataFrame({f'gp_mae{gp_j}': mae})
                                df_gp_pct = pd.DataFrame({f'gp_pct{gp_j}': pct_inside}, index=[0])
                                df_gp_nlpd = pd.DataFrame({f'gp_nlpd{gp_j}': nlpd})
                                
                            else:
                                new_df = pd.DataFrame({f'gp_mae{gp_j}': mae})
                                df_gp = pd.concat([df_gp, new_df], axis=1)
                                
                                new_df = pd.DataFrame({f'gp_pct{gp_j}': pct_inside}, index=[0])
                                df_gp_pct = pd.concat([df_gp_pct, new_df], axis=1)

                                new_df = pd.DataFrame({f'gp_nlpd{gp_j}': nlpd})
                                df_gp_nlpd = pd.concat([df_gp_nlpd, new_df], axis=1)
                            
                            gp_j += 1
                            psd_error = False
                        except:
                            psd_error = True
                            print(f'Not PSD Error, jitter of {jitter} added without success')

                elif model == 'multitask_gp':

                    if torch.isnan(y_train).any():
                        # remove rows with nans
                        mask = ~torch.isnan(y_train).any(axis=1)
                        y_train = y_train[mask]
                        x_train = x_train[mask]
                    
                    if torch.isnan(y_test).any():
                        # remove rows with nans
                        mask = ~torch.isnan(y_test).any(axis=1)
                        y_test = y_test[mask]
                        x_test = x_test[mask]
                    
                    likelihood = MultitaskBetaLikelihood(
                        num_tasks=y_train.size(-1),
                        scale=25,
                    )
                    mean, covar = get_mean_covar(num_latent=8)
                    model = MultitaskGPModel(
                        X=x_train,
                        y=y_train,
                        likelihood=likelihood,
                        mean_module=mean,
                        covar_module=covar,
                        num_latents=8,
                        jitter=1e-2,
                    )
                    try:
                        model.fit(n_iter=200, lr=0.1, verbose=True, use_wandb=False)
                        pred_dist = model.predict(x_test)
                        y_pred, lower, upper = model.predict(x_test, pred_type='median')

                        nlpd = NLPD(pred_dist, y_test).median(axis=0).values
                        
                        for k in range(y_train.size(-1)):
                            pred = y_pred[:,k]
                            y_t = y_test[:,k]
                            low = lower[:,k]
                            up = upper[:,k]
                    
                            pct_inside = inside_ci(low, up, y_t.numpy())
                            mae = mean_absolute_error(y_t, pred)
                        
                            if df_multitask_gp is None:
                                df_multitask_gp = pd.DataFrame({f'multitask_gp_mae{gp_j}': mae})
                                df_multitask_gp_pct = pd.DataFrame({f'multitask_gp_pct{gp_j}': pct_inside}, index=[0])
                                df_multitask_nlpd = pd.DataFrame({f'multitask_gp_nlpd{gp_j}': nlpd[:, k]})
                            
                            else:
                                new_df = pd.DataFrame({f'multitask_gp_mae{gp_j}': mae})
                                df_multitask_gp = pd.concat([df_multitask_gp, new_df], axis=1)
                                
                                new_df = pd.DataFrame({f'multitask_gp_pct{gp_j}': pct_inside}, index=[0])
                                df_multitask_gp_pct = pd.concat([df_multitask_gp_pct, new_df], axis=1)

                                new_df = pd.DataFrame({f'multitask_gp_nlpd{gp_j}': nlpd[:, k]})
                                df_multitask_nlpd = pd.concat([df_multitask_nlpd, new_df], axis=1)
                            
                            multitask_j += 1

                    except:
                        print(f'Not PSD Error, jitter of {jitter} added without success')
                    

                elif model == 'holt':
                    # TODO 
                    pass
    

        # dictionary of data frames
        df_dict = {'VAR': df_var, 
                'Seasonal Exponential Smoothing': df_exp, 
                'Simple Exponential Smoothing': df_simple_exp, 
                'Yesterday': df_yesterday, 
                'Hourly Average': df_hourly_avg, 
                'Persistence': df_persistence,
                'GP': df_gp,
                'GP NLPD': df_gp_nlpd,
                }

        # save data frames to csv in current directory
        for model, df in df_dict.items():
            df.to_csv(f'df_{model}_{season}.csv')

        
        mean_var, lower_var, upper_var = get_mean_ci(df_var)
        mean_exp, lower_exp, upper_exp = get_mean_ci(df_exp)
        mean_exp_nlpd, lower_exp_nlpd, upper_exp_nlpd = get_mean_ci(df_exp_nlpd)
        mean_simple_exp, lower_simple_exp, upper_simple_exp = get_mean_ci(df_simple_exp)
        mean_simple_exp_nlpd, lower_simple_exp_nlpd, upper_simple_exp_nlpd = get_mean_ci(df_simple_exp_nlpd)
        mean_yesterday, lower_yesterday, upper_yesterday = get_mean_ci(df_yesterday)
        mean_hourly_avg, lower_hourly_avg, upper_hourly_avg = get_mean_ci(df_hourly_avg)
        mean_persistence, lower_persistence, upper_persistence = get_mean_ci(df_persistence)
        mean_gp, lower_gp, upper_gp = get_mean_ci(df_gp)
        mean_nlpl_gp, lower_nlpl_gp, upper_nlpl_gp = get_mean_ci(df_gp_nlpd)
        mean_multi_gp, lower_multi_gp, upper_multi_gp = get_mean_ci(df_multitask_gp)
        mean_nlpl_multi_gp, lower_nlpl_multi_gp, upper_nlpl_multi_gp = get_mean_ci(df_multitask_nlpd)



        results = {'VAR': {'mean': mean_var, 'lower': lower_var, 'upper': upper_var},
        'Seasonal Exponential Smoothing': {'mean': mean_exp, 'lower': lower_exp, 'upper': upper_exp},
        'Simple Exponential Smoothing': {'mean': mean_simple_exp, 'lower': lower_simple_exp, 'upper': upper_simple_exp},
        'Yesterday': {'mean': mean_yesterday, 'lower': lower_yesterday, 'upper': upper_yesterday},
        'Hourly Average': {'mean': mean_hourly_avg, 'lower': lower_hourly_avg, 'upper': upper_hourly_avg},
        'Persistence': {'mean': mean_persistence, 'lower': lower_persistence, 'upper': upper_persistence},
        'GP': {'mean': mean_gp, 'lower': lower_gp, 'upper': upper_gp},
        'Multitask GP': {'mean': mean_multi_gp, 'lower': lower_multi_gp, 'upper': upper_multi_gp},
        }

        # save results to csv in current directory
        for model_name, result_dict in results.items():
            df = pd.DataFrame(result_dict)
            df.to_csv(f'results_{model_name}_{season}.csv')
        
        nlpds = {'GP': {'mean': mean_nlpl_gp, 'lower': lower_nlpl_gp, 'upper': upper_nlpl_gp},
            'Multitask GP': {'mean': mean_nlpl_multi_gp, 'lower': lower_nlpl_multi_gp, 'upper': upper_nlpl_multi_gp},
            'Seasonal Exponential Smoothing': {'mean': mean_exp_nlpd, 'lower': lower_exp_nlpd, 'upper': upper_exp_nlpd},
            'Simple Exponential Smoothing': {'mean': mean_simple_exp_nlpd, 'lower': lower_simple_exp_nlpd, 'upper': upper_simple_exp_nlpd},
            }
        
        for model_name, result_dict in nlpds.items():
            df = pd.DataFrame(result_dict)
            df.to_csv(f'nlpd_{model_name}_{season}.csv')
    
    if plot_results:
        plot_forecast_mae(results, season)
        boxplot_models(results, season)
        boxplot_forecast_horizon(df_dict, season=season)
