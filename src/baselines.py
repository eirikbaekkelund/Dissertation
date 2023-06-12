from statsmodels.tsa.api import AutoReg, ARIMA, SARIMAX, ExponentialSmoothing, MarkovAutoregression

class ExponentialModels(ExponentialSmoothing):
    """ 
    Global class for exponential smoothing models

    Args:
        time_train (torch.Tensor): training time
        y_train (torch.Tensor): training data
        time_test (torch.Tensor): test time
        y_test (torch.Tensor): test data
    """
    def __init__(self, y_train, y_test, trend):
        super(ExponentialModels, self).__init__(y_train, trend=trend)
        self.y_train = y_train
        self.y_test = y_test
    
    def single_exponential_smoothing(self, alpha : float = 0.2):
        """
        Single exponential smoothing model
        
        Args:
            alpha (float): smoothing parameter
        """
        self.fit(smoothing_level=alpha, optimized=False)
        return self.forecast(len(self.y_test))
    
    def double_exponential_smoothing(self, alpha : float = 0.2, beta : float = 0.2, trend : str = 'add'):
        """
        Double exponential smoothing model
        
        Args:
            alpha (float): smoothing parameter for level
            beta (float): smoothing parameter for trend
            trend (str): type of trend to use
        """
        assert trend in ['add', 'mul'], 'trend must be either add or mul'

        self.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False, use_brute=True)
        return self.forecast(len(self.y_test))

# TODO: Add more models for baselines