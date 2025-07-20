# simulate_with_mpc.py

import numpy as np
from nominal_mpc import solve_nominal_mpc
from tube_mpc import TubeMPC
from constraints import tighten_constraints
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.highway_env.vehicle_models import UnicycleModel

model = UnicycleModel(dt=0.12)

# MPC parameters
horizon = 15
Q = np.diag([20.0, 20.0, 5.0])
R = np.diag([0.01, 0.01])
Qf = np.diag([50.0, 50.0, 20.0])
u_bounds = (np.array([-2.0, -1.0]), np.array([2.0, 1.0]))
x_bounds = (np.array([-1e6, -1e6, -1e6]), np.array([1e6, 1e6, 1e6]))
disturbance_bound = np.array([0.0,0.0,0.0])

# Setup Tube MPC
tube_mpc = TubeMPC(model, horizon, Q, R, Qf, u_bounds, x_bounds, disturbance_bound)

# Initial state
x = np.array([0.0, 0.0, 0.0])
trajectory = [x.copy()]
target= np.array([0.5,1.0])

# Run for 50 steps
for step in range(50):

    theta_desired= np.arctan2(target[1]-x[1], target[0]-x[0])
    target_traj = np.tile(np.array([target[0], target[1], theta_desired]), (horizon + 1, 1))

    u_nom_seq, x_nom_seq = tube_mpc.plan(x, target_traj)

    if u_nom_seq is None:
        print("MPC failed to solve.")
        break

    # Apply first control input
    u = u_nom_seq[:,0]
    x = model.step(x, u)
    trajectory.append(x)

    if np.linalg.norm(x[:2] - target[:2]) < 0.02 and abs(x[2] - theta_desired) < 0.1:
        print("Reached the goal.")
        break

trajectory = np.array(trajectory)

start = np.array([0.0, 0.0])
end= np.array([target[0], target[1]])

# Plot
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory")
plt.plot([start[0], end[0]], [start[1], end[1]], 'r--', label="Target")
plt.legend()
plt.grid()
plt.savefig("nominal_mpc_single_vehicle.png")