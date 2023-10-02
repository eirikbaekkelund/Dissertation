import torch
import gpytorch
import wandb
import numpy as np
from gpytorch.variational import (CholeskyVariationalDistribution, 
                                  MeanFieldVariationalDistribution,
                                  TrilNaturalVariationalDistribution,
                                  NaturalVariationalDistribution,
                                  VariationalStrategy,
                                  IndependentMultitaskVariationalStrategy,
                                  LMCVariationalStrategy)
from torch.utils.data import TensorDataset, DataLoader
from data.utils import store_gp_module_parameters
from typing import Optional

class HadamardGPModel(gpytorch.models.ApproximateGP):
    """
    Class for creating a Hadamard GP model.
    If latents are passed, then the model is a LCM model.
    Otherwise, the model is a multitask model with independent multitask variational inference.
    The type of variational distribution is specified by the variational_type argument.

    Args:
        X (torch.Tensor): training data
        y (torch.Tensor): training labels
        mean_module (gpytorch.means.Mean): mean function
        covar_module (gpytorch.kernels.Kernel): covariance function
        likelihood (gpytorch.likelihoods.Likelihood): likelihood function
        num_tasks (int): number of tasks
        learn_inducing_locations (bool, optional): whether to learn inducing locations. Defaults to True.
        num_latents (Optional[int], optional): number of latent functions. Defaults to 1.
        variational_type (str, optional): type of variational distribution. Defaults to "cholesky".
        jitter (Optional[float], optional): jitter value. Defaults to None.
    """
    def __init__(self, 
        X : torch.Tensor,
        y : torch.Tensor,
        mean_module : gpytorch.means.Mean,
        covar_module : gpytorch.kernels.Kernel,
        likelihood : gpytorch.likelihoods.Likelihood,
        num_tasks : int,
        learn_inducing_locations : bool = True,
        num_latents : Optional[int] = None,
        variational_type : str = "cholesky",
        inducing_proportion : Optional[float] = None,
        jitter : Optional[float] = None
    ):

        # need to ensure that y_train is in (0, 1)
        if y.max() >= 1:
            y[y >= 1] = 1 - 1e-4
        if y.min() <= 0:
            y[y <= 0] = 1e-4
        
        if learn_inducing_locations:
            assert inducing_proportion is not None, "Must specify inducing_proportion when learning inducing locations"
            n_inducing = int(X.shape[0] * inducing_proportion)
            idx = np.random.choice(X.shape[0], n_inducing, replace=False)
            inducing_points = X[idx, :]
            
        else:
            inducing_points = X
        
        # mps is currently not supported in GPyTorch or PyTorch due to Cholesky decomposition not being supported
        # if torch.backends.mps.is_available():
        #     inducing_points.to('mps')
        
        if num_latents is not None:
            assert num_latents > 1, "Must specify num_latents > 1 when using LMC"
          
        variational_strategy = self.get_strategy(
            variational_type=variational_type,
            inducing_points=inducing_points,
            num_tasks=num_tasks,
            num_latents=num_latents,
            jitter=jitter
        )
        
        super().__init__(variational_strategy)
        
        self.X = X
        self.y = y
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood
        self.variational_type = variational_type
        self.num_tasks = num_tasks
        self.jitter = jitter

    def forward(self, x):
     
        mean = self.mean_module(x)
        covar = self.covar_module(x) + self.jitter * torch.eye(x.size(0))
    
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    def get_variational_distribution(self, 
            variational_type : str,
            num_inducing_points : int,
            batch_shape : torch.Size,
            jitter : Optional[float] = None
        ):
        """ 
        Creates a variational distribution based on the type of variational distribution specified.
        This is for the approximation of the posterior distribution of the latent function.

        Args:
            variational_type (str): type of variational distribution
            num_inducing_points (int): number of inducing points
            batch_shape (torch.Size): batch shape of the variational distribution
            jitter (Optional[float], optional): jitter value. Defaults to None.
        
        Returns:
            gpytorch.variational.VariationalDistribution: variational distribution
        """
        
        if variational_type == "cholesky":
            return CholeskyVariationalDistribution(
                num_inducing_points=num_inducing_points,
                batch_shape=batch_shape,
                jitter=jitter
            )
        
        elif variational_type == 'mean_field':
            return MeanFieldVariationalDistribution(
                num_inducing_points=num_inducing_points,
                batch_shape=batch_shape,
                jitter=jitter
            )
        
        elif variational_type == 'tril_natural':
            return TrilNaturalVariationalDistribution(
                num_inducing_points=num_inducing_points,
                batch_shape=batch_shape,
                jitter=jitter
            )
        
        elif variational_type == 'natural':
            return NaturalVariationalDistribution(
                num_inducing_points=num_inducing_points,
                batch_shape=batch_shape,
                jitter=jitter
            )
        
        else:
            raise ValueError(f"Variational type {variational_type} not recognized")

    def get_strategy(self,
        variational_type : str,
        inducing_points : torch.Tensor,
        learn_inducing_locations : bool = True,
        num_tasks : int = 1,
        num_latents : Optional[int] = 1,
        jitter : Optional[float] = None
    ):
        """
        Creates a variational strategy based on the type of variational distribution specified.
        This is for the model's variational inference. Either an independent multitask variational
        strategy or a LMC variational strategy is created.

        Args:
            variational_type (str): type of variational distribution
            inducing_points (torch.Tensor): inducing points
            learn_inducing_locations (bool, optional): whether to learn inducing locations. Defaults to True.
            num_tasks (int, optional): number of tasks. Defaults to 1.
            num_latents (Optional[int], optional): number of latent functions. Defaults to 1.
            jitter (Optional[float], optional): jitter value. Defaults to None.
        
        Returns:
            gpytorch.variational.VariationalStrategy: variational strategy
        """
        assert variational_type in ["cholesky", "mean_field", "tril_natural", "natural"], "Variational type not recognized must be one of 'cholesky', 'mean_field', 'tril_natural', 'natural'"
        if num_latents is None:
            variational_distribution = self.get_variational_distribution(
                variational_type=variational_type,
                num_inducing_points=inducing_points.size(0),
                batch_shape=torch.Size([num_tasks]),
                jitter=jitter
            )
            return IndependentMultitaskVariationalStrategy(
                VariationalStrategy(
                    self,
                    inducing_points=inducing_points,
                    variational_distribution=variational_distribution,
                    learn_inducing_locations=learn_inducing_locations,
                    jitter_val=jitter
                ),
                num_tasks=num_tasks,
            )
        
        elif num_latents > 1:
            variational_distribution = self.get_variational_distribution(
                variational_type=variational_type,
                num_inducing_points=inducing_points.size(0),
                batch_shape=torch.Size([num_latents]),
                jitter=jitter
            )
            return LMCVariationalStrategy(
                VariationalStrategy(
                    self,
                    inducing_points=inducing_points,
                    variational_distribution=variational_distribution,
                    learn_inducing_locations=learn_inducing_locations,
                    jitter_val=jitter
                ),
                num_tasks=num_tasks,
                num_latents=num_latents,
                jitter_val=jitter
            )
        else:
            raise ValueError("num_latents must be greater than 1 if LMC otherwise num_latents must be None")
    
    def train_full(self,
        n_iter : int,
        task_indices : torch.Tensor,
        optimizer : list,
        elbo : gpytorch.mlls.VariationalELBO,
        verbose : bool = False,
        use_wandb : bool = False
    ):
        """
        Fits the model using full batch variational inference.
        """
   
        print_freq = 10
        j = 0
        for i in range(n_iter):
            # zero grad all optimizers in optimizer list
            [opt.zero_grad() for opt in optimizer]
            output = self(self.X, task_indices=task_indices)
            loss = -elbo(output, self.y, task_indices=task_indices)
            loss.backward()
            [opt.step() for opt in optimizer]

            self.losses.append(loss.item())
                            
            if verbose and (i + 1) % print_freq == 0:
                print(f"Iter {i + 1}/{n_iter} - Loss: {loss.item()}")

            if use_wandb and i % print_freq == 0:
                pass 
                # TODO log to wandb
            
            if i > 0:
                if abs(self.losses[-2] - self.losses[-1]) < 1e-3:
                    j += 1
                    if j == 15:
                        print(f'Early stopping at iteration {i}')
                        break
                else:
                    j = 0

    
    def train_minibatch(self,
        n_iter : int,
        task_indices : torch.Tensor,
        optimizer : list,
        elbo : gpytorch.mlls.VariationalELBO,
        verbose : bool = False,
        use_wandb : bool = False
    ):
        """
        Fits the model using minibatch variational inference.
        Minibatch proved to not work well with the Hadamard kernel.
        """
       
        print_freq = 10
        
        dataset = TensorDataset(self.X, self.y, task_indices)
        dataloader = DataLoader(dataset, batch_size=500, shuffle=True)

        for i in range(n_iter):
            for x_batch, y_batch, task_batch in dataloader:
                [opt.zero_grad() for opt in optimizer]
                output = self(x_batch, task_indices=task_batch)
                loss = -elbo(output, y_batch, task_indices=task_batch)
                loss.backward()
                [opt.step() for opt in optimizer]
            
            self.losses.append(loss.item())
            
            if verbose and (i + 1) % print_freq == 0:
                print(f"Iter {i + 1}/{n_iter} - Loss: {loss.item()}")

            if use_wandb and i % print_freq == 0:
                log_dict = store_gp_module_parameters(self)
                log_dict['loss'] = loss.item()
                wandb.log(log_dict)
        
    def fit(self,
        n_iter : int,
        lr : float,
        task_indices : torch.Tensor,
        verbose : bool = False,
        use_wandb : bool = False,
        minibatch : bool = False
    ):
        self.train()
        self.likelihood.train()
        # use natural gradients if tril_natural or natural
        if self.variational_type in ['tril_natural', 'natural']:
            optimizer = [gpytorch.optim.NGD(self.variational_parameters(), num_data=self.y.size(0), lr=lr),
                         torch.optim.Adam(self.parameters(), lr=lr*0.1)]
        else:
            optimizer = [torch.optim.Adam(self.parameters(), lr=lr)]

        elbo = gpytorch.mlls.VariationalELBO(likelihood=self.likelihood,
                                             model=self,
                                             num_data=self.y.size(0))
        
        if use_wandb:
            wandb.init(
                project="hadamard-gp",
                config={'lr': lr, 'n_iter': n_iter, }
            )
        self.losses = []
        # seems to do poorly with minibatch - unstable mixing of tasks
        if minibatch:
            self.train_minibatch(n_iter, task_indices, optimizer, elbo, verbose, use_wandb)
        else:
            self.train_full(n_iter, task_indices, optimizer, elbo, verbose, use_wandb)
        
        if use_wandb:
            wandb.finish()
    
    def get_i_prediction(self, i, task_indices):
        assert i < self.num_tasks, "i must be less than the number of tasks"
        if not hasattr(self, 'y_pred'):
            raise ValueError("y_pred must be computed before calling predict_i. Call predict() first.")

        return (self.y_pred[task_indices == i], 
               self.lower[task_indices == i], 
               self.upper[task_indices == i])
    
    def predict_dist(self):
        # check that y_pred has been computed
        if not hasattr(self, 'y_pred'):
            raise ValueError("y_pred must be computed before calling predict_i. Call predict() first.")
        
        return self.dist
    
    def predict(self, 
        x : torch.Tensor, 
        task_indices : torch.Tensor,
        num_samples : int = 100):
        
        self.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            self.dist = self.likelihood(self(x, task_indices=task_indices), task_indices=task_indices)
        
        samples = self.dist.sample(sample_shape=torch.Size([num_samples]))

        for idx in torch.unique(task_indices):
            # slice samples by task
            samples_idx = samples[:,:, task_indices == idx]

            # get MC estimate of mean and variance
            y_pred_idx = samples_idx.median(dim=0).values.mean(dim=0)
            lower_idx = np.quantile(samples_idx.numpy(), 0.025, axis=0).mean(axis=0)
            upper_idx = np.quantile(samples_idx.numpy(), 0.975, axis=0).mean(axis=0)
            
            # get all the predictions in the same order as the input
            if idx == 0:
                y_pred = y_pred_idx
                lower = lower_idx
                upper = upper_idx
            else:
                y_pred = np.concatenate((y_pred, y_pred_idx), axis=-1)
                lower = np.concatenate((lower, lower_idx), axis=-1)
                upper = np.concatenate((upper, upper_idx), axis=-1)
        
        self.y_pred = y_pred
        self.lower = lower
        self.upper = upper
        
        return self.y_pred, self.lower, self.upper
    
    def set_gpu(self):
        """
        Checks if cuda is available and moves model and the 
        training data to cuda if available.
        """
        if torch.cuda.is_available():
            print("Using CUDA")
            self.cuda()
            self.likelihood.cuda()
            self.X = self.X.cuda()
            self.y = self.y.cuda()
        elif torch.backends.mps.is_available():
            print("Using MPS")
            device = torch.device("mps")
            self.to(device)
            self.likelihood.to(device)
            self.X = self.X.to(device)
            self.y = self.y.to(device)
    
    def set_cpu(self):
        """
        Moves model and the training data to cpu.
        """
        print("Using CPU")
        self.cpu()
        self.likelihood.cpu()
        self.X = self.X.cpu()
        self.y = self.y.cpu()
    
    def warm_start(self, model_dict):
        """
        Warm starts the model with the parameters in model_dict.
        """
        try:
            self.load_state_dict(model_dict)
            try:
                # load all parameters except base_variational_strategy variational parameters as they are not compatible
                # due to mismatch in number of training points              
                for name, param in model_dict.items():
                    if 'base_variational_strategy' not in name:
                        self.state_dict[name].copy_(param)
            except Exception as e:
                print(e)
                
        except Exception as e:
            print(e)
    
    def posterior(self, x, task_indices):
        """ 
        Returns the samples from the posterior distribution of the latent function.
        """

        

