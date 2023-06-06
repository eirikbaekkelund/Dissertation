import matplotlib.pyplot as plt
import numpy as np

def plot_grid(df, COORDS, RADIUS):
    """
    Plot the grid of the UK with the PV systems
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df['longitude_noisy'], df['latitude_noisy'], alpha=0.2, color='b', label='PV systems')
    ldn_circle = plt.Circle((COORDS[1], COORDS[0]), RADIUS, color='r', fill=True, alpha=0.2, label='Area')
    ax.add_patch(ldn_circle)

    # TODO add uk map background


    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('PV systems in the UK')
    ax.legend()
    
    plt.show()


def plot_gp(pred_train, pred_test, time_train, time_test, y_train, y_test):
    
    # plot predictions
    plt.figure(figsize=(13, 5))

    plt.scatter(time_train, y_train, marker='x', color='black', alpha=0.7, label='Observed Data')
    plt.scatter(time_test, y_test, marker='x', color='black', alpha=0.7)

    plt.plot(time_train, pred_train.mean, color='r', label='Mean')

    lower, upper = pred_train.confidence_region()
    # clip the lower and upper confidence region to avoid negative values and greater than 1
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    plt.fill_between(time_train, lower, upper, alpha=0.4, color='pink', label='Confidence')



    plt.plot(time_test, pred_test.mean, color='r')
    lower, upper = pred_test.confidence_region()

    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)

    plt.fill_between(time_test, lower, upper, alpha=0.4, color='pink')
    plt.vlines(x= len(time_train), ymin=0, ymax=max(y_train.max(), y_test.max()), 
            color='black', linestyle='--', label='Train-Test Split')


    plt.legend()

    plt.xlabel('Time (5 min intervals between 8am and 4pm)', fontsize=13)
    plt.ylabel('PV Production (0-1 Scale)')

    plt.show();