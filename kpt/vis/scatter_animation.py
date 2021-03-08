from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def scatter_scene(array, file_name):
    fig = plt.figure()
    ax = Axes3D(fig)

    # Initialize scatters
    scatters = [ax.scatter([x], [y], [z]) for x, y, z in array]
    # ax.scatter(array[:,0], array[:,1], array[:,2])
    ax.set_xlabel('X')
    ax.set_xlim(-30,30)
    ax.set_ylabel('Y')
    ax.set_ylim(-30,30)
    ax.set_zlabel('Z')
    ax.set_zlim(-30,30)
    ax.view_init(110, -90)
    plt.savefig('test_image.png')

def animate_scatters(frame, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.

    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)

    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[frame][i,0:1], data[frame][i,1:2], data[frame][i,2:])
    return scatters

def scatter_animation(array, file_name):
    """
    array: sequence X joints X xyz_value

    """
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = Axes3D(fig)

    # Initialize scatters
    init_frame = array[0]
    init_coord = init_frame.reshape(-1, 3)
    scatters = [ax.scatter([x], [y], [z]) for x, y, z in init_coord]

    # Number of iterations
    iterations = array.shape[0]

    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_xlim(-20,20)
    ax.set_ylabel('Y')
    ax.set_ylim(-10,25)
    ax.set_zlabel('Z')
    ax.set_zlim(-20,20)

    ax.set_title('Frame')

    # Provide starting angle for the view.
    ax.view_init(110, -90)

    ani = animation.FuncAnimation(fig, animate_scatters, frames=iterations, fargs=(array.reshape(array.shape[0],-1,3), scatters),
                                    interval=100)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, bitrate=200, extra_args=['-vcodec', 'libx264'])
    ani.save(file_name, writer=writer)