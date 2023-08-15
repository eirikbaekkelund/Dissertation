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
    If latents are passed, then the model is a LMC model.
    Otherwise, the model is a multitask model.
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
        
        if torch.backends.mps.is_available():
            inducing_points.to('mps')
        
        if num_latents is not None:
            assert num_latents > 1, "Must specify num_latents > 1 when using LMC"
          
            variational_strategy = self.get_strategy(
                variational_type=variational_type,
                inducing_points=inducing_points,
                num_tasks=num_tasks,
                num_latents=num_latents,
                jitter=jitter
            )
        else:
            variational_strategy = self.get_strategy(
                variational_type=variational_type,
                inducing_points=inducing_points,
                num_tasks=num_tasks,
                jitter=jitter
            )
            

        super().__init__(variational_strategy)
        
        self.X = X
        self.y = y
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood
        self.variational_type = variational_type        

    def forward(self, x):
     
        mean = self.mean_module(x)
        covar = self.covar_module(x)
    
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
        # TODO natural gradients
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

    def predict(self, 
        x : torch.Tensor, 
        task_indices : torch.Tensor,
        num_samples : int = 100):
        
        self.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            preds = self.likelihood(self(x, task_indices=task_indices), task_indices=task_indices)
            samples = preds.sample(sample_shape=torch.Size([num_samples]))
        
        y_pred = torch.zeros(x.size(0), task_indices.max() + 1, device=x.device)
        lower = torch.zeros(x.size(0), task_indices.max() + 1, device=x.device)
        upper = torch.zeros(x.size(0), task_indices.max() + 1, device=x.device)
        
        for idx in torch.unique(task_indices):
            
            samples_idx = samples[:,:, task_indices == idx]
            y_pred_idx = samples_idx.mean(dim=0).median(dim=0).values
            lower_idx = np.quantile(samples_idx.numpy(), 0.025, axis=0).mean(axis=0)
            upper_idx = np.quantile(samples_idx.numpy(), 0.975, axis=0).mean(axis=0)
            
            y_pred[:, idx] = y_pred_idx
            lower[:, idx] = lower_idx
            upper[:, idx] = upper_idx
        
        return y_pred, lower, upper
    
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
        

    
    

