import torch
import gpytorch
import numpy as np
import pandas as pd
import time
from models import (ApproximateGPBaseModel,
                    MultitaskGPModel,
                    HadamardGPModel,
                    LSTM, 
                    fit_bayesian_ridge,
                    fit_xgboost)
from models.baselines_temporal import (YesterdayForecast, 
                              HourlyAverage, 
                              Persistence, 
                              fit_var, 
                              fit_exp, 
                              fit_simple_exp,
                              var_exp_simulation)

from metrics import (mean_absolute_error, 
                     inside_ci, 
                     nlpd_holt, 
                     neg_log_pred,
                     neg_log_pred_hadamard)
from kernels import get_mean_covar_weather, get_mean_covar
from likelihoods import (BetaLikelihood_MeanParametrization, 
                         MultitaskBetaLikelihood,
                         HadamardBetaLikelihood)
from data import SystemLoader, PVWeatherGenerator

np.random.seed(42)
torch.manual_seed(42)


"""
Apologies to anyone who reads this code. It is a mess.
Time pressure and the need to get results quickly led to this :(
"""


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
    CIRCLE_COORDS = (55.1, -3.05)
    RADIUS = 0.25
    MODELS_EXG = ['LSTM', 'BayesianRidge', 'XGBoost', 'SimpleGP', 'HadamardGP']
    MODELS_TEMP = ['Persistence', 'Yesterday', 'Hourly', 'VAR', 
                'ES', 'Seasonal ES', 'Simple GP', 'LCM MT-GP']
    # set models to be the list of models you want to run
    MODELS = MODELS_TEMP + MODELS_EXG

    jitter = 1e-4
    gp_config = {
        'type' : 'stochastic', # SVI
        'name' : 'cholesky', # type of posterior covariance approximation
        'jitter' : jitter, # jitter for numerical stability
    }
    mean, quasi_periodic = get_mean_covar()
    constant_mean = gpytorch.means.ConstantMean(batch_shape=torch.Size([1]))
    gp_inputs ={
        'jitter': jitter,
        'learn_inducing_locations': False,
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
    num_latents = N_SYSTEMS // 2 + 1 # (6 // 2 + 1 = 4)
    interval = 6 # for inducing points of hadamard GP

    train_interval = int(DAILY_DATA_POINTS * N_DAYS_FOLD) # number of data points per system in each fold
    # create loader that will iterate over the data
    loader = SystemLoader(df, train_interval=train_interval, n_hours_pred=6)

    psd_error = False
    psd_hadamard_error = False

    # this could be passed as a dict by running over the models
    gp_temp_j = 0
    lstm_j = 0
    brr_j = 0
    xgb_j = 0
    hadamard_j = 0
    var_j = 0
    exp_j = 0
    simple_exp_j = 0
    yesterday_j = 0
    hourly_avg_j = 0
    persist_j = 0
    gp_j = 0
    multitask_j = 0

    # this too can be passed as a dict of dicts
    brr, xgb, gp, hadamard, lstm = {}, {}, {}, {}, {}
    brr_nlpd, gp_nlpd, hadamard_nlpd = {}, {}, {}
    gp_pct, hadamard_pct = {}, {}
    vareg = {}
    exp, exp_nlpd = {}, {}
    simple_exp, simple_exp_nlpd = {}, {}
    yday = {}
    hourly = {}
    persistence = {}
    gp_temp, gp_temp_pct, gp_temp_nlpd = {}, {}, {}
    multitask_gp, multitask_gp_pct, multitask_gp_nlpd = {}, {}, {}
    psd_error = False

    # wandb.init(project='dissertation', entity='dissertation', config={'models': MODELS})
    

    for i, (X_tr, Y_tr, X_te, Y_te, T_tr, T_te) in enumerate(loader):

        print(f'Fold {i+1} of {len(loader)}\n')
        
        if X_tr.shape[0] < 100 or X_tr[T_tr == 0].shape[0] < 100:
            break
        
        if 'HadamardGP' in MODELS:
            try:
                mean, covar = get_mean_covar_weather(num_latents=num_latents, 
                                                    d=X_tr.shape[1], 
                                                    weather_kernel='matern',
                                                    use_ard_dim=True)
                
                model_hadamard = HadamardGPModel(
                    X=X_tr[::interval],
                    y=Y_tr[::interval],
                    mean_module=mean,
                    covar_module=covar,
                    likelihood=HadamardBetaLikelihood(num_tasks=num_tasks, scale=15),
                    num_tasks=num_tasks,
                    num_latents=num_latents,
                    learn_inducing_locations=False,
                    inducing_proportion=1,
                    jitter=jitter,
                )
                try:
                    model_hadamard.warm_start(had_state_dict)
                except:
                    pass
                start_had = time.time()
                model_hadamard.fit(n_iter=250, lr=0.2, task_indices=T_tr[::interval], verbose=True)
                had_state_dict = model_hadamard.state_dict
                end_had = time.time()
                model_hadamard.predict(X_te, T_te)
                pred_dist_hadamard = model_hadamard.predict_dist()
                nlpd = neg_log_pred_hadamard(pred_dist_hadamard, Y_te)
                nlpd[(nlpd == torch.inf) | (nlpd == -torch.inf)] = torch.nan
                nlpd_hadamard = nlpd.median(axis=-1).values
                print('-*-'*10)
                print(f'Time Hadamard: {end_had-start_had:.3f} (s)')

                for t in range(num_tasks):
                    y_pred_had, lower_had, upper_had = model_hadamard.get_i_prediction(t, T_te)
                    _,_,_,y_te = loader.train_test_split_individual(t)
                    pct_inside_had = inside_ci(lower_had, upper_had, y_te.numpy())
                    mae_had = mean_absolute_error(y_pred_had, y_te.numpy())
                    nlpd_had = nlpd_hadamard[T_te == t].numpy()
                    nlpd_had = nlpd_had * len(nlpd_hadamard) / len(nlpd_had)

                    print(f'HadamardGP: {hadamard_j + 1}')
                    print(f'Avg MAE: {mae_had.mean():.3f}', f'Avg NLPD: {np.nanmean(nlpd_had):.3f}', f'Pct Inside: {pct_inside_had}')
                    print('-*-*'*10)

                    hadamard[f'MAE_{hadamard_j}'] = mae_had
                    hadamard_nlpd[f'NLPD_{hadamard_j}'] = nlpd_had
                    hadamard_pct[f'PCT_{hadamard_j}'] = pct_inside_had
                    
                    hadamard_j += 1
            
            except Exception as e:
                # set nan arrays of the prediction length
                hadamard[f'mae_{hadamard_j}'] = np.full(len(Y_te), np.nan)
                hadamard_nlpd[f'nlpd_{hadamard_j}'] = np.full(len(Y_te), np.nan)
                hadamard_pct[f'pct_{hadamard_j}'] = np.full(len(Y_te), np.nan)
                hadamard_j += 1
                print(e)
        
        if 'LCM MT-GP' in MODELS:

            y_train = torch.stack([Y_tr[T_tr == t] for t in range(num_tasks)], dim=-1).float()
            y_test = torch.stack([Y_te[T_te == t] for t in range(num_tasks)], dim=-1).float()
            x =  torch.linspace(0, 100, len(y_train) + len(y_test), dtype=torch.float32)
            x_train, x_test = x[:len(y_train)], x[len(y_train):]
            mean, covar = get_mean_covar(num_latent=num_latents, base_kernel='matern')
            model = MultitaskGPModel(
                X=x_train,
                y=y_train,
                likelihood= MultitaskBetaLikelihood(scale=5, num_tasks=y_train.size(-1)),
                mean_module=mean if (i < 9 and i > 35) else gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents])),
                covar_module=covar,
                num_latents=num_latents,
                variational_dist='cholesky',
                learn_inducing_locations=True,
                jitter=1e-2,
            )
            try:
                try:
                    model.warm_start(mt_state_dict)
                except:
                    pass
                model.fit(n_iter=250, lr=0.2, verbose=True, use_wandb=False)           
                print('Final dispersion: ', model.likelihood.scale)

                mt_state_dict = model.state_dict
                pred_dist = model.predict(x_test)
                y_pred, lower, upper = model.predict(x_test, pred_type='median')

                nlpd = neg_log_pred(pred_dist, y_test).median(axis=0).values / num_tasks
                
                for k in range(y_train.size(-1)):
                    pred = y_pred[:,k]
                    y_t = y_test[:,k]
                    low = lower[:,k]
                    up = upper[:,k]
            
                    pct_inside = inside_ci(low, up, y_t.numpy())
                    mae = mean_absolute_error(y_t, pred)

                    multitask_gp[f'Kroenecker_mae{multitask_j}'] = mae
                    multitask_gp_pct[f'Kroenecker_pct{multitask_j}'] = pct_inside
                    multitask_gp_nlpd[f'Kroenecker_nlpd{multitask_j}'] = nlpd[:,k] 
                
                    multitask_j += 1
                
                    print(f'KroeneckerGP {multitask_j} | Avg MAE: {mae.mean():.3f}, Avg NLPD: {np.nanmean(nlpd[:,k]):.3f}, Pct Inside: {pct_inside}')
                    print('-*-'*10)

            except Exception as e:
                print('Multitask GP error:')
                print(e)
                # set nan arrays of the prediction length
                arr = np.nan * np.ones(len(y_test))
                for k in range(y_train.size(-1)):
                    multitask_gp[f'Kroenecker_mae{gp_j}'] = arr
                    multitask_gp_pct[f'Kroenecker_pct{gp_j}'] = arr
                    multitask_gp_nlpd[f'Kroenecker_nlpd{gp_j}'] = arr
                    multitask_j += 1

        if 'VAR' in MODELS:
            y_train = np.stack([Y_tr[T_tr == t] for t in range(num_tasks)], axis=-1)
            y_test = np.stack([Y_te[T_te == t] for t in range(num_tasks)], axis=-1)
            y_pred = fit_var(y_train, len(y_test))
            mae = mean_absolute_error(y_test, y_pred)
            for t in range(num_tasks):
                vareg[f'VAR_mae{var_j}'] = mae[:,t]
                var_j += 1
             
            

        for s in range(loader.n_systems):
            x_tr, y_tr, x_te, y_te = loader.train_test_split_individual(s)
            
            for m in MODELS:
                if m == 'LSTM':
                    model_lstm = LSTM(
                        x_train=x_tr[:,:-1].float(), # last column is the time index
                        y_train=y_tr.float(),
                        hidden_units=1,
                        n_layers=3,
                        dropout=0.14,
                        batch_size=40,
                    )
                    start = time.time()
                    model_lstm.fit(n_iter=150, lr=2e-3)
                    end = time.time()
                    y_pred = model_lstm.predict(x_te[:,:-1].float(), y_te.float())
                    mae = mean_absolute_error(y_pred, y_te.numpy())
                    print('-*-*'*10)
                    print(f'LSTM: {lstm_j+1} | Time: {end-start:.3f} (s)')
                    print(f'Avg MAE: {mae.mean():.3f}')
                    print('-*-*'*10)
                    lstm[f'MAE_{lstm_j}'] = mae
                    lstm_j += 1

                elif m == 'BayesianRidge':
                    start = time.time()
                    y_pred, var = fit_bayesian_ridge(x_tr[:,:-1], y_tr, x_te[:,:-1])
                    end = time.time()
                    mae = mean_absolute_error(y_pred, y_te.numpy())
                    nlpd_brr = nlpd_holt(y_pred, var, y_te.numpy())
                    
                    print(f'BRR: {brr_j+1} | Time: {end-start:.3f} (s)')
                    print(f'Avg MAE: {mae.mean():.3f}, Avg NLPD: {nlpd_brr.mean()}')
                    print('-*-*'*10)
                    brr[f'MAE_{brr_j}'] = mae
                    brr_nlpd[f'NLPD_{brr_j}'] = nlpd_brr
                    brr_j += 1
                
                elif m == 'XGBoost':
                    start = time.time()
                    y_pred = fit_xgboost(x_tr, y_tr, x_te[:,:-1])
                    end = time.time()
                    mae = mean_absolute_error(y_pred, y_te.numpy())
                    print(f'XGBoost: {xgb_j + 1} | Time: {end-start:.3f} (s)')
                    print(f'Avg MAE: {mae.mean():.3f}')
                    print('-*-*'*10)
                    xgb[f'MAE_{xgb_j}'] = mae
                    xgb_j += 1

                elif m == 'SimpleGP':
                    gp_config['num_inducing_points'] = x_tr.shape[0]
                    mean, covar = get_mean_covar_weather(num_latents=1, 
                                                        d=x_tr.shape[1], 
                                                        weather_kernel='matern',
                                                        use_ard_dim=True)
                    model_gp = ApproximateGPBaseModel(
                            X=x_tr,
                            y=y_tr,
                            mean_module=mean,
                            covar_module=covar,
                            likelihood=BetaLikelihood_MeanParametrization(scale=15),
                            config=gp_config)
                    
                    try:
                        start = time.time()
                        try:
                            model_gp.warm_start(gp_state_dict)
                        except:
                            pass
                        
                        model_gp.fit(n_iter=250, lr=0.2, verbose=True)
                        end = time.time()
                        gp_state_dict = model_gp.state_dict()
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

                        gp[f'MAE_{gp_j}'] = mae
                        gp_nlpd[f'NLPD_{gp_j}'] = nlpd
                        gp_pct[f'PCT_{gp_j}'] = pct_inside
                   
                        gp_j += 1
                        psd_error = False

                    except Exception as e:
                        arr = np.zeros_like(y_te)
                        arr[:] = np.nan
                        gp[f'mae_{gp_j}'] = arr
                        gp_nlpd[f'nlpd_{gp_j}'] = arr
                        gp_pct[f'pct_{gp_j}'] = arr
                        gp_j += 1
                        print(e)

                elif m == 'Seasonal ES':
                    y_pred, fitted_model = fit_exp(y_tr.numpy(), len(y_te))
                    es_var = var_exp_simulation(fitted_model, y_pred, n_pred=len(y_te))
                    
                    mae = mean_absolute_error(y_te.numpy(), y_pred)
                    nlpd = nlpd_holt(y_pred, es_var, y_te.numpy())
                    exp[f'Seasonal ES_mae{exp_j}'] = mae
                    exp_nlpd[f'Seasonal ES_nlpd{exp_j}'] = nlpd
                    print('Seasonal ES mae: ', mae.mean())
                    print('-*-'*10)

                    exp_j += 1
                
                elif m == 'ES':
                    y_pred, fitted_model = fit_simple_exp(y_tr.numpy(), len(y_te))
                    es_var = var_exp_simulation(fitted_model, y_pred, n_pred=len(y_te))
                
                    mae = mean_absolute_error(y_te.numpy(), y_pred)
                    nlpd = nlpd_holt(y_pred, es_var, y_te.numpy())
                    simple_exp[f'Simple ES_mae{simple_exp_j}'] = mae
                    simple_exp_nlpd[f'Simple ES_nlpd{simple_exp_j}'] = nlpd

                    print('Simple ES mae: ', mae.mean())
                    print('-*-'*10)

                    simple_exp_j += 1
                
                elif m == 'Yesterday':
                    model = YesterdayForecast(DAY_MIN, DAY_MAX, MINUTE_INTERVAL)
                    y_pred = model.predict(y_tr)
                    mae = mean_absolute_error(y_te, y_pred)

                    print('Yesterday mae: ', mae.mean())
                    print('-*-'*10)
                    
                    yday[f'Yesterday_mae{yesterday_j}'] = mae
                    yesterday_j += 1
                
                elif m == 'Hourly':
                    model = HourlyAverage(DAY_MIN, DAY_MAX, MINUTE_INTERVAL)
                    y_pred = model.predict(y_tr)
                    # if y_te is one dim, make y_pred one dim
                    if len(y_te.shape) == 1:
                        y_pred = y_pred.squeeze(-1)
                    
                    mae = mean_absolute_error(y_te.numpy(), y_pred)

                    print('Hourly mae: ', mae.mean())
                    print('-*-'*10)

                    hourly[f'Hourly Avg_mae{hourly_avg_j}'] = mae
                    hourly_avg_j += 1
                
                elif m == 'Persistence':
                    model = Persistence()
                    y_pred = model.predict(y_tr, len(y_te))
                    mae = mean_absolute_error(y_te.squeeze(-1).numpy(), y_pred)

                    print('Persistence mae: ', mae.mean())
                    print('-*-'*10)
                
                    persistence[f'Persistence_mae{persist_j}'] = mae
                    persist_j += 1
                
                elif m == 'Simple GP': 
                    mean, covar = get_mean_covar(num_latent=1)         
                    gp_config['num_inducing_points'] = x_tr.shape[0]
                    gp_inputs['config'] = gp_config
                    gp_inputs['likelihood'] = BetaLikelihood_MeanParametrization(scale=15)
                    gp_inputs['mean_module'] = mean
                    gp_inputs['covar_module'] = covar
                    gp_inputs['X'] = x_tr[:,-1]
                    gp_inputs['y'] = y_tr

                    model = ApproximateGPBaseModel(**gp_inputs)
                    try:
                        model.fit(n_iter=200, lr=0.2, use_wandb=False, verbose=True)
                        gp_temp_state_dict = model.state_dict()
                        y_pred, lower, upper = model.predict(x_te[:,-1], pred_type='median')
                        pred_dist = model.predict(x_te[:,-1], pred_type='dist')
                        
                        pct_inside = inside_ci(lower, upper, y_te.numpy())
                        mae = mean_absolute_error(y_te, y_pred)
                        nlpd = neg_log_pred(pred_dist, y_te).median(axis=0).values

                        gp_temp[f'Simple GP_mae{gp_temp_j}'] = mae
                        gp_temp_pct[f'Simple GP_pct{gp_temp_j}'] = pct_inside
                        gp_temp_nlpd[f'Simple GP_nlpd{gp_temp_j}'] = nlpd
                        gp_temp_j += 1

                        print(f'Simple GP: {gp_temp_j} | Avg MAE: {mae.mean():.3f}, Avg NLPD: {np.nanmean(nlpd):.3f}, Pct Inside: {pct_inside}')
                        print('-*-'*10)
                
                    except Exception as e:
                        # set nan arrays of the prediction length
                        arr = np.zeros_like(y_te)
                        arr[:] = np.nan
                        gp_temp[f'Simple GP_mae{gp_temp_j}'] = arr
                        gp_temp_pct[f'Simple GP_pct{gp_temp_j}'] = arr
                        gp_temp_nlpd[f'Simple GP_nlpd{gp_temp_j}'] = arr
                        gp_temp_j += 1
                        print(e)
                

    model_dict = {'VAR': vareg, 
            'SeasonalES': exp,
            'Seasonal ES_nlpd': exp_nlpd,
            'Simple ES': simple_exp,
            'Simple ES_nlpd': simple_exp_nlpd,
            'Yesterday': yday,
            'Hourly Average': hourly,
            'Persistence': persistence,
            'SimpleGP': gp,
            'SimpleGP_pct': gp_pct,
            'Simple GP_nlpd': gp_nlpd,
            'KroneckerGP': multitask_gp,
            'KroneckerGP_pct': multitask_gp_pct,
            'KroneckerGP_nlpd': multitask_gp_nlpd,
            }
    
    # save results in a csv file where each column is an iteration and should have rows equal to the length of the test array
    
    
    for model in model_dict.keys():
        try:
            df = pd.DataFrame.from_dict(model_dict[model])        
        except Exception as e:
            print(e)
            try:
                df = pd.DataFrame(model_dict[model], index=[0])
            except Exception as e:
                print(e)
                print(model)
        
        df.to_csv(f'{model}.csv')
    
    
    # save all results as dataframes then save them as csv files
    dict_models = {
        'LSTM' : lstm,
        'BRR' : brr,
        'BRR_nlpd' : brr_nlpd,
        'XGBoost' : xgb,
        'SimpleGP' : gp,
        'SimpleGP_nlpd' : gp_nlpd,
        'SimpleGP_pct' : gp_pct,
        'HadamardGP' : hadamard,
        'HadamardGP_nlpd' : hadamard_nlpd,
        'HadamardGP_pct' : hadamard_pct,
    }
    for k, v in dict_models.items():
        try:
            df = pd.DataFrame(v)
        except ValueError:
            df = pd.DataFrame(v, index=[0])

        df.to_csv(f'{k}.csv')
