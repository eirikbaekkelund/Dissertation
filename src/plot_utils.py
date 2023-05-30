import matplotlib.pyplot as plt

def plot_grid(df, COORDS, RADIUS):
    """
    Plot the grid of the UK with the PV systems
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df['longitude_noisy'], df['latitude_noisy'], alpha=0.2, color='b', label='PV systems')
    ldn_circle = plt.Circle((COORDS[1], COORDS[0]), RADIUS, color='r', fill=True, alpha=0.2, label='London Area')
    ax.add_patch(ldn_circle)

    # TODO add uk map background


    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('PV systems in the UK')
    ax.legend()
    
    plt.show()