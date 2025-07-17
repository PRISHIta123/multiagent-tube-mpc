import numpy as np
import matplotlib.pyplot as plt
from env.highway_env.vehicle_models import UnicycleModel

def rollout_unicycle(initial_state, control_sequence, dt=0.1):
    """
    Rolls out a trajectory using the UnicycleModel.
    Returns the list of visited states.
    """
    model = UnicycleModel(dt=dt)
    trajectory = model.simulate_trajectory(initial_state, control_sequence)
    return trajectory

def plot_trajectory(trajectory, title="Trajectory", save_path=None):
    """
    Plots a 2D trajectory.
    """
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'b-o', label="Trajectory")
    plt.plot(x[0], y[0], 'go', label="Start")
    plt.plot(x[-1], y[-1], 'ro', label="End")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
