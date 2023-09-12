
import torch
import numpy as np
import pandas as pd
import time
from models import (ApproximateGPBaseModel,
                    HadamardGPModel,
                    LSTM, 
                    fit_bayesian_ridge,
                    fit_xgboost)
from metrics import (mean_absolute_error, 
                     inside_ci, 
                     nlpd_holt, 
                     get_mean_ci,
                     neg_log_pred,
                     neg_log_pred_hadamard)
from kernels import get_mean_covar_weather
from pv_plot import plot_forecast_mae, boxplot_forecast_horizon, boxplot_models
from likelihoods import BetaLikelihood_MeanParametrization, HadamardBetaLikelihood
from data import SystemLoader, PVWeatherGenerator

np.random.seed(42)
torch.manual_seed(42)


def get_beta(idx):
    # winter
    return 15
    # if idx <= 9 or idx > 48:
    #     return 30
    # # otherwise more fluctuations in PV, so assign greater variance by lower
    # # dispersion parameter
    # else:
    #     return 15


if __name__ == "__main__":
    # data parameters (Data is from Manchester Area)
    DAY_INIT = 0 # start of data is 01-01-2018
    N_DAYS = 365 # data until 01-01-2019
    DAY_MIN = 8 # starting hour of all our data (per day)
    DAY_MAX = 16 # ending hour of all our data (per day)
    MINUTE_INTERVAL = 5 # the minute interval of our data
    DAILY_DATA_POINTS = (DAY_MAX - DAY_MIN) * 60 // MINUTE_INTERVAL
    N_DAYS_FOLD = 7
    N_SYSTEMS = 6
    CIRCLE_COORDS = (53.28, -3.05)
    RADIUS = 0.25
    MODELS = ['LSTM', 'BayesianRidge', 'XGBoost', 'SimpleGP', 'HadamardGP']
    jitter = 1e-5
    gp_config = {
        'type' : 'stochastic', # SVI
        'name' : 'cholesky', # type of posterior covariance approximation
        'jitter' : jitter, # jitter for numerical stability
    }
  
    # generator takes data from the entire data base and filters out what we specify 
    generator = PVWeatherGenerator(
        coords=CIRCLE_COORDS,
        radius=RADIUS,
        day_init=DAY_INIT,
        n_systems=N_SYSTEMS,
        n_days=N_DAYS,
        minute_interval=MINUTE_INTERVAL,
    )
    df = generator.df
    num_tasks = N_SYSTEMS
    num_latents = N_SYSTEMS // 2 + 1
    interval = 12 # for inducing points of hadamard GP

    train_interval = int(DAILY_DATA_POINTS * N_DAYS_FOLD) # number of data points per system in each fold
    # create loader that will iterate over the data
    loader = SystemLoader(df, train_interval=train_interval)

    psd_error = False
    psd_hadamard_error = False

    gp_j = 0
    lstm_j = 0
    brr_j = 0
    xgb_j = 0
    hadamard_j = 0

    df_gp = None
    df_lstm = None
    df_brr = None
    df_xgb = None
    df_hadamard = None

    for i, (X_tr, Y_tr, X_te, Y_te, T_tr, T_te) in enumerate(loader):
        beta_scale = get_beta(i)
        print(f'Fold {i+1} of {len(loader)}\n')
        if X_tr.shape[0] < 100 or X_tr[T_tr == 0].shape[0] < 100:
            break
        try:
            mean, covar = get_mean_covar_weather(num_latents=num_latents, d=X_tr.shape[1])
            model_hadamard = HadamardGPModel(
                X=X_tr[::interval],
                y=Y_tr[::interval],
                mean_module=mean,
                covar_module=covar,
                likelihood=HadamardBetaLikelihood(num_tasks=num_tasks, scale=beta_scale),
                num_tasks=num_tasks,
                num_latents=num_latents,
                learn_inducing_locations=False,
                inducing_proportion=1,
                jitter=jitter,
            )
            model_hadamard.set_cpu()
            start_had = time.time()
        
            model_hadamard.fit(n_iter=200, lr=0.2, task_indices=T_tr[::interval])
            end_had = time.time()
            model_hadamard.predict(X_te, T_te)
            pred_dist_hadamard = model_hadamard.predict_dist()
            nlpd = neg_log_pred_hadamard(pred_dist_hadamard, Y_te)
            nlpd[(nlpd == torch.inf) | (nlpd == -torch.inf)] = torch.nan
            nlpd_hadamard = nlpd.median(axis=-1).values
            print('-*-'*10)
            print('Beta scale: ', beta_scale)
            print(f'Time Hadamard: {end_had-start_had:.3f} (s)')
            psd_hadamard_error = False
        except Exception as e:
            psd_hadamard_error = True
            print(e)
           
        for i in range(loader.n_systems):
            x_tr, y_tr, x_te, y_te = loader.train_test_split_individual(i)
            
            for m in MODELS:
                if m == 'LSTM':
                    model_lstm = LSTM(
                        x_train=x_tr.float(), # last column is the time index
                        y_train=y_tr.float(),
                        hidden_units=1,
                        n_layers=3,
                        dropout=0.14,
                        batch_size=40,
                    )
                    start = time.time()
                    model_lstm.fit(n_iter=150, lr=2e-3)
                    end = time.time()
                    y_pred = model_lstm.predict(x_te.float(), y_te.float())
                    mae = mean_absolute_error(y_pred, y_te.numpy())
                    print('-*-*'*10)
                    print(f'LSTM: {lstm_j+1} | Time: {end-start:.3f} (s)')
                    print(f'Avg MAE: {mae.mean():.3f}')
                    print('-*-*'*10)
                    if df_lstm is None:
                        df_lstm = pd.DataFrame({f'{m}_mae{lstm_j}' : mae})
                    else:
                        new_df = pd.DataFrame({f'{m}_mae{lstm_j}' : mae})
                        df_lstm = pd.concat([df_lstm, new_df], axis=1)
                    lstm_j += 1

                elif m == 'BayesianRidge':
                    start = time.time()
                    y_pred, var = fit_bayesian_ridge(x_tr, y_tr, x_te)
                    end = time.time()
                    lower_brr = y_pred - 1.96 * np.sqrt(var)
                    upper_brr = y_pred + 1.96 * np.sqrt(var)
                    mae = mean_absolute_error(y_pred, y_te.numpy())
                    nlpd_brr = nlpd_holt(y_pred, var, y_te.numpy())
                    
                    print(f'BRR: {brr_j+1} | Time: {end-start:.3f} (s)')
                    print(f'Avg MAE: {mae.mean():.3f}, Avg NLPD: {nlpd_brr.mean()}')
                    print('-*-*'*10)

                    if df_brr is None:
                        df_brr = pd.DataFrame({f'{m}_mae{brr_j}' : mae})
                        df_brr_nlpd = pd.DataFrame({f'{m}_{brr_j}_nlpd' : nlpd_brr})
                    else:
                        new_df = pd.DataFrame({f'{m}_mae{brr_j}' : mae})
                        df_brr = pd.concat([df_brr, new_df], axis=1)
                        new_df = pd.DataFrame({f'{m}_nlpd{brr_j}' : nlpd_brr})
                        df_brr_nlpd = pd.concat([df_brr_nlpd, new_df], axis=1)

                    brr_j += 1
                
                elif m == 'XGBoost':
                    start = time.time()
                    y_pred = fit_xgboost(x_tr, y_tr, x_te,
                                         max_depth=8, max_leaves=7, n_estimators=400)
                    end = time.time()
                    mae = mean_absolute_error(y_pred, y_te.numpy())
                    print(f'XGBoost: {xgb_j + 1} | Time: {end-start:.3f} (s)')
                    print(f'Avg MAE: {mae.mean():.3f}')
                    print('-*-*'*10)
                    if df_xgb is None:
                        df_xgb = pd.DataFrame({f'{m}_mae{xgb_j}' : mae})
                    else:
                        new_df = pd.DataFrame({f'{m}_mae{xgb_j}' : mae})
                        df_xgb = pd.concat([df_xgb, new_df], axis=1)
                    xgb_j += 1

                
                elif m == 'SimpleGP':
                    gp_config['num_inducing_points'] = x_tr.shape[0]
                    mean, covar = get_mean_covar_weather(num_latents=1, d=x_tr.shape[1])
                    model_gp = ApproximateGPBaseModel(
                            X=x_tr,
                            y=y_tr,
                            mean_module=mean,
                            covar_module=covar,
                            likelihood=BetaLikelihood_MeanParametrization(scale=beta_scale),
                            config=gp_config)
                    
                    if i != 0 and not psd_error:
                        model_gp.warm_start(state_dict)
                    try:
                        start = time.time()
                        model_gp.fit(n_iter=100, lr=0.2, verbose=False)
                        end = time.time()
                        state_dict = model_gp.state_dict()
                        y_pred, lower_gp, upper_gp = model_gp.predict(x_te, pred_type='median')
                        pred_dist_gp = model_gp.predict(x_te, pred_type='dist')
                            
                        pct_inside = inside_ci(lower_gp, upper_gp, y_te.numpy())
                        mae = mean_absolute_error(y_te, y_pred)
                        nlpd = neg_log_pred(pred_dist_gp, y_te).median(axis=0).values
                        nlpd[nlpd.isinf()] = np.nan
                        print('-*-*'*10)
                        print(f'SimpleGP: {gp_j + 1} | Time: {end-start:.3f} (s)')
                        print(f'Avg MAE: {mae.mean():.3f}', f'Avg NLPD: {np.nanmean(nlpd):.3f}', f'Pct Inside: {pct_inside}')
                        print('-*-*'*10)
                        
                        if df_gp is None:
                            df_gp = pd.DataFrame({f'{m}_mae{gp_j}': mae})
                            df_gp_pct = pd.DataFrame({f'{m}_pct{gp_j}': pct_inside}, index=[0])
                            df_gp_nlpd = pd.DataFrame({f'{m}_nlpd{gp_j}': nlpd})
                            
                        else:
                            new_df = pd.DataFrame({f'{m}_mae{gp_j}': mae})
                            df_gp = pd.concat([df_gp, new_df], axis=1)
                            new_df = pd.DataFrame({f'{m}_pct{gp_j}': pct_inside}, index=[0])
                            df_gp_pct = pd.concat([df_gp_pct, new_df], axis=1)
                            new_df = pd.DataFrame({f'{m}_nlpd{gp_j}': nlpd})
                            df_gp_nlpd = pd.concat([df_gp_nlpd, new_df], axis=1)
                        
                        gp_j += 1
                        psd_error = False
                    except Exception as e:
                        print(e)
                        psd_error = True
                
                elif m == 'HadamardGP' and not psd_hadamard_error:
                    y_pred_had, lower_had, upper_had = model_hadamard.get_i_prediction(i, T_te)
                    pct_inside_had = inside_ci(lower_had, upper_had, y_te.numpy())
                    mae_had = mean_absolute_error(y_pred_had, y_te.numpy())
                    nlpd_had = nlpd_hadamard[T_te == i].numpy()
                    # rescale nlpd_had 
                    nlpd_had = nlpd_had * len(nlpd_hadamard) / len(nlpd_had)

                    print(f'HadamardGP: {hadamard_j + 1}')
                    print('-*-*'*10)
                    print(f'Avg MAE: {mae_had.mean():.3f}', f'Avg NLPD: {np.nanmean(nlpd_had):.3f}', f'Pct Inside: {pct_inside_had}')
                    print('-*-*'*10)

                    if df_hadamard is None:
                        df_hadamard = pd.DataFrame({f'{m}_mae{hadamard_j}': mae_had})
                        df_hadamard_pct = pd.DataFrame({f'{m}_pct{hadamard_j}': pct_inside_had}, index=[0])
                        df_hadamard_nlpd = pd.DataFrame({f'{m}_{hadamard_j}': nlpd_had})
                    else:
                        new_df = pd.DataFrame({f'{m}_mae{hadamard_j}': mae_had})
                        df_hadamard = pd.concat([df_hadamard, new_df], axis=1)
                        new_df = pd.DataFrame({f'{m}_pct{hadamard_j}': pct_inside_had}, index=[0])
                        df_hadamard_pct = pd.concat([df_hadamard_pct, new_df], axis=1)
                        new_df = pd.DataFrame({f'{m}_{hadamard_j}': nlpd_had})
                        df_hadamard_nlpd = pd.concat([df_hadamard_nlpd, new_df], axis=1)
                    
                    hadamard_j += 1
            print('\n')
    
    df_nlpd = {
        'BRR': df_brr_nlpd,
        'SimpleGP': df_gp_nlpd,
        'HadamardGP': df_hadamard_nlpd
    }
    for model, df in df_nlpd.items():
        df.to_csv(f'{model}_nlpd.csv')
    
    df_pct = {
        'SimpleGP': df_gp_pct,
        'HadamardGP': df_hadamard_pct
    }
    for model, df in df_pct.items():
        df.to_csv(f'{model}_pct.csv')

    df_dict = {
        'BRR': df_brr,
        'XGBoost': df_xgb,
        'SimpleGP': df_gp,
        'HadamardGP': df_hadamard,
        'LSTM': df_lstm
    }

    for model, df in df_dict.items():
        df.to_csv(f'{model}.csv')
    
    mean_brr, lower_var, upper_var = get_mean_ci(df_brr)
    mean_xgb, lower_var, upper_var = get_mean_ci(df_xgb)
    mean_gp, lower_var, upper_var = get_mean_ci(df_gp)
    mean_had, lower_var, upper_var = get_mean_ci(df_hadamard)
    mean_lstm, lower_var, upper_var = get_mean_ci(df_lstm)
    
    mean_brr_nlpd, lower_brr_nlpd, upper_brr_nlpd = get_mean_ci(df_brr_nlpd)
    mean_gp_nlpd, lower_gp_nlpd, upper_gp_nlpd = get_mean_ci(df_gp_nlpd)
    mean_had_nlpd, lower_had_nlpd, upper_had_nlpd = get_mean_ci(df_hadamard_nlpd)
    mean_gp_pct, lower_gp_pct, upper_gp_pct = get_mean_ci(df_gp_pct)
    mean_had_pct, lower_had_pct, upper_had_pct = get_mean_ci(df_hadamard_pct)

    results = {
        'BRR': {'mean': mean_brr, 'lower': lower_var, 'upper': upper_var},
        'XGBoost': {'mean': mean_xgb, 'lower': lower_var, 'upper': upper_var},
        'SimpleGP': {'mean': mean_gp, 'lower': lower_var, 'upper': upper_var},
        'HadamardGP': {'mean': mean_had, 'lower': lower_var, 'upper': upper_var},
        'LSTM': {'mean': mean_lstm, 'lower': lower_var, 'upper': upper_var},
        'BRR_nlpd': {'mean': mean_brr_nlpd, 'lower': lower_brr_nlpd, 'upper': upper_brr_nlpd},
        'GP_nlpd': {'mean': mean_gp_nlpd, 'lower': lower_gp_nlpd, 'upper': upper_gp_nlpd},
        'HadamardGP_nlpd': {'mean': mean_had_nlpd, 'lower': lower_had_nlpd, 'upper': upper_had_nlpd},
        'GP_pct': {'mean': mean_gp_pct, 'lower': lower_gp_pct, 'upper': upper_gp_pct},
        'HadamardGP_pct': {'mean': mean_had_pct, 'lower': lower_had_pct, 'upper': upper_had_pct}
    }


    for model_name, results_dict in results.items():
        df = pd.DataFrame(results_dict)
        df.to_csv(f'{model_name}_results.csv')
    
    plot_results = True
    if plot_results:
        plot_forecast_mae(results, pred_points = mean_brr.shape[0])
        boxplot_models(results)
        boxplot_forecast_horizon(results)
    
                


                