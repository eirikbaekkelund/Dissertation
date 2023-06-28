from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution, UnwhitenedVariationalStrategy
from gpytorch.variational import NaturalVariationalDistribution, TrilNaturalVariationalDistribution

#############################################
######## Variational Distribution ###########
#############################################

class VariationalBase:
    """ 
    Class for creating variational distributions based on specifications 
    in the configuration dictionary to be used in the variational 
    strategy for the GP model.

    Args:
        config (dict): configuration of the GP model
    """
    def __init__(self, config):
        # check for gradient type
        if config['type'] == 'stochastic':
            self.variational_distribution = self.get_stochastic(config)
        elif config['type'] == 'natural':
            self.variational_distribution = self.get_natural(config)
    
    def delete_non_config(self, config):
        """ 
        Delete non-configuration parameters

        Args:
            config (dict): configuration of the GP model
        """
        
        _config = {key : value for key, value in config.items() if key not in ['type', 'name']}
        
        return _config
        
    def get_stochastic(self, config):
        """ 
        Create stochastic variational distribution

        Args:
            config (dict): configuration of the GP model
        """
        assert config['name'] in ['cholesky', 'mean_field'], 'Variational distribution must be either cholesky or mean_field'
        
        name = config['name']
        config = self.delete_non_config(config)

        if name == 'cholesky':
            return CholeskyVariationalDistribution(**config)
        elif name == 'mean_field':
            return MeanFieldVariationalDistribution(**config)
    
    def get_natural(self, config):
        """ 
        Create natural variational distribution

        Args:
            config (dict): configuration of the GP model
        """
        assert config['name'] in ['natural', 'tril_natural'], 'Variational distribution must be either natural or tril_natural'
        
        name = config['name']
        config = self.delete_non_config(config)

        if name == 'natural':
            return NaturalVariationalDistribution(**config)
        elif name == 'tril_natural':
            return TrilNaturalVariationalDistribution(**config)
