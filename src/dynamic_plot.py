import matplotlib.pyplot as plt
import torch
import imageio.v2 as imageio
import os 

def mean_median_beta(alpha, beta):
    """
    Compute the mean and median of a beta distribution
    given the alpha and beta parameters.

    Args:
        alpha (float): Alpha parameter
        beta (float): Beta parameter
    
    Returns:
        mean (float): Mean of the beta distribution
        median (float): Median of the beta distribution
    """
    mean = alpha / (alpha + beta)
    median = (alpha - 1/3) / (alpha + beta - 2/3)

    if median < 0:
        median = 0
    
    elif median > 1:
        median = 1
    
    return mean, median

def plot_dynamic_beta_dist(title, save_path):
    """ 
    Create a dynamic simulation video for a beta distribution
    where the x value is changing over time.

    Args:
        title (str): Title of the plot
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        legend (list): List of strings for the legend
        save_path (str): Path to save the plot
    
    Saves a ffmpeg video of the plot.
    
    """

    x = torch.linspace(0, 1, 1000)
    
    dispersion_parameter = torch.linspace(5, 90, 3)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))


    # Plot the distribution over changing x vals
    for d in dispersion_parameter:
        for xi in x:
            alpha = xi * d
            beta = d - alpha
            if alpha <= 0 or beta <= 0:
                continue
            y = torch.distributions.beta.Beta(alpha, beta).log_prob(x).exp()
            mean, median = mean_median_beta(alpha, beta)

            plt.plot(x, y, color='blue', alpha=0.7)
            plt.axvline(mean, color='red', linestyle='--', alpha=0.6, label='Mean')
            plt.axvline(median, color='green', linestyle='--', alpha=0.6, label='Median')
            
            plt.legend(loc='upper left')
            plt.title(title + f', Dispersion = {d:.2f}')
            
            plt.savefig(f'{save_path}beta_{d}_{xi:.4}.png')

            plt.cla()

def create_gif(path, gif_name):
    """ 
    Create a gif from a series of images.
    
    Args:
        path (str): Path to the images
        gif_name (str): Name of the gif
    
    Saves a gif of the images.
    
    """
    # get files in path
    files = os.listdir(path)
    # sort files by number after underscore then by number after second underscore
    # delete any files in the directory that are not images

    files = [file for file in files if file.endswith('.png')]
    files.sort(key=lambda x: (float(x.split('_')[1]), float(x.split('_')[2].split('.')[0])))

    images = []
    for filename in files:
        images.append(imageio.imread(path + filename))
    
    imageio.mimsave(gif_name, images, duration=100)

if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(__file__)) + '/beta_dist_pics/'
    #plot_dynamic_beta_dist('Beta Distribution', path)
    create_gif(path, 'beta_dist.gif')
