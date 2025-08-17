import matplotlib.pyplot as plt
import time
import numpy as np


def visualize(simulation, frame_length=0.2):
    """
    Shows a simulation of touchscreen movement

    simulation: A list of numpy arrays, the output of the create_touch function from the simulator
    frame_length: How long each frame is displayed
    """
    plt.ion()  # turn on interactive mode, non-blocking `show`
    test = plt.imshow(simulation[0], cmap="CMRmap_r", interpolation="nearest", vmax=4)
    for frame in simulation:
        test.set_data(frame)
        plt.pause(frame_length)
        time.sleep(frame_length)
    plt.close()
