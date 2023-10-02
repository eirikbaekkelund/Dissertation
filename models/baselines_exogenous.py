import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from torch.utils.data import DataLoader
from data import SequenceDataset

def fit_bayesian_ridge(x_train, y_train, x_test):
    """
    Fit a Bayesian Ridge regression model to the data
    """
    if isinstance(x_train, torch.Tensor):
        x_train = x_train.numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(x_test, torch.Tensor):
        x_test = x_test.numpy()

    brr = BayesianRidge(n_iter=10).fit(x_train, y_train)
    y_pred, std = brr.predict(x_test, return_std=True)
    var = std**2
    y_pred = np.clip(y_pred, 0, 1)
    # where y_pred - 2*std < 0, set variance so that y - 2*std = 0
    var[y_pred - 2*std < 0] = y_pred[y_pred - 2*std < 0] / 2
    var[y_pred + 2*std > 1] = (1 - y_pred[y_pred + 2*std > 1]) 
    return y_pred, var

def fit_xgboost(x_train, 
                y_train, 
                x_test,
                max_depth=3,
                max_leaves=3,
                n_estimators=150,):
    """
    Fit an XGBoost model to the data
    """
    if isinstance(x_train, torch.Tensor):
        x_train = x_train.numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(x_test, torch.Tensor):
        x_test = x_test.numpy()

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", 
                                 max_depth=max_depth,
                                 max_leaves=max_leaves,
                                 n_estimators=n_estimators,
                                 n_jobs=-1,
                                 random_state=42)
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    y_pred = np.clip(y_pred, 0, 1)
    return y_pred

class LSTM(nn.Module):
    """ 
    An LSTM for time series forecasting.
    
    Args:
        n_features (int): The number of features in the time series dataset.
        hidden_units (int): The number of hidden units in the LSTM layer.
        n_layers (int): The number of LSTM layers.
    """
    def __init__(self,
        x_train : torch.Tensor,
        y_train : torch.Tensor,
        hidden_units : int = 10,
        n_layers : int = 1,
        dropout : float = 0.1,
        batch_size : int = 10,
    ):
        super().__init__()
        self.train_dataset = SequenceDataset(x_train, y_train, sequence_length=30)
        # find batch size divisor
        
        self.batch_size = self.get_batch_size(x_train, batch_size)
        self.hidden_units = hidden_units
        self.num_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size=x_train.shape[-1],
            hidden_size=hidden_units,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
        )
        
        self.linear = nn.Linear(hidden_units, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        
        out = self.linear(hn[0]).flatten()
        # sigmoid to ensure output is between 0 and 1
        torch.sigmoid(out)
        return out
    
    def get_batch_size(self, x, batch_size=16):
        while len(x) % batch_size != 0:
            batch_size += 1
            
            if batch_size > len(x):
                batch_size = len(x)
                break
        
        return min(batch_size, len(x))

    
    def predict(self, x, y, batch_size : int = 8):
        """
        Predict the output of the LSTM model on the given dataset.

        Args:
            loader (DataLoader): The DataLoader for the dataset.
        
        Returns:
            torch.Tensor: The predictions of the LSTM model.
        """
        dataset = SequenceDataset(x, y)        
        x, _ = next(iter(dataset))
        batch_size = self.get_batch_size(x, batch_size=batch_size)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.eval()
        output = torch.tensor([])
        with torch.no_grad():
            for x, _ in loader:
                y_pred = self(x)
                output = torch.cat((output, y_pred),0)
        torch.clamp(output, min=0, max=1, out=output)
        return output
    
    def fit(self,
        n_iter : int = 100,
        lr : float = 0.01,
        verbose : bool = False,
    ):
        """
        Fit the LSTM model to the training data.

        Args:
            n_iter (int): The number of iterations to train the model.
            lr (float): The learning rate.
            verbose (bool): Whether to print the training loss.
        """
        
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        print_freq = n_iter // 10
        
        for epoch in range(n_iter):
            total_loss = 0
            for x, y in loader:
                optimizer.zero_grad()
                y_pred = self(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % print_freq == 0:
                print(f'Epoch {epoch + 1}/{n_iter} | Loss: {(total_loss / len(loader)):.4f}')

