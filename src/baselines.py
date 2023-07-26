from statsmodels.tsa.api import (AutoReg, 
                                 ARIMA, 
                                 SARIMAX, 
                                 ExponentialSmoothing, 
                                 SimpleExpSmoothing,
                                 MarkovAutoregression)

from abc import ABC, abstractmethod

class TimeSeriesBase(ABC):
    """
    Base class for exponential smoothing models
    """
    def __init__(self, y_train, y_test):
        # 
        self.y_train = y_train
        self.y_test = y_test
    
    def predict(self, **kwargs):
        return self.model.fit(**kwargs).forecast(steps=len(self.y_test))


class ExponentialSmoothingModel(TimeSeriesBase):
    """
    Exponential smoothing model
    """
    def __init__(self, y_train, y_test, **kwargs):
        super().__init__(y_train, y_test)
        self.model = ExponentialSmoothing(endog=self.y_train, **kwargs)

class SimpleExpSmoothingModel(TimeSeriesBase):
    """
    Simple exponential smoothing model
    """
    def __init__(self, y_train, y_test, **kwargs):
        super().__init__(y_train, y_test)
        self.model = SimpleExpSmoothing(endog=self.y_train, **kwargs)


class AutoRegressionModel(TimeSeriesBase):
    """
    Auto regression model
    """
    def __init__(self, y_train, y_test, **kwargs):
        super().__init__(y_train, y_test)
        self.model = AutoReg(endog=self.y_train, **kwargs)


class ARIMAModel(TimeSeriesBase):
    """
    ARIMA model
    """
    def __init__(self, y_train, y_test, **kwargs):
        super().__init__(y_train, y_test)
        self.model = ARIMA(endog=self.y_train, **kwargs)


class SARIMAXModel(TimeSeriesBase):
    """
    SARIMAX model
    """
    def __init__(self, y_train, y_test, **kwargs):
        super().__init__(y_train, y_test)
        self.model = SARIMAX(endog=self.y_train, **kwargs)


class MarkovAutoregressionModel(TimeSeriesBase):
    """
    Markov autoregression model
    """
    def __init__(self, y_train, y_test, **kwargs):
        super().__init__(y_train, y_test)
        self.model = MarkovAutoregression(endog=self.y_train, **kwargs)


