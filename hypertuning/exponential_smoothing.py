from hypertuning.base import HyperOptBase
import optuna

class ExpSmoothingHyperOpt(HyperOptBase):
    """ 
    Hyperparameter optimization using Optuna for an exponential smoothing model.
    """
    def sample_params(self, trial: optuna.trial.Trial):
        """
        Sample hyperparameters for the model.
        """
        smoothing_level = trial.suggest_float('smoothing_level', 0.0, 1.0)
        smoothing_slope = trial.suggest_float('smoothing_slope', 0.0, 1.0)
        damping_slope = trial.suggest_float('damping_slope', 0.0, 1.0) 
        return smoothing_level, smoothing_slope, damping_slope
    
    