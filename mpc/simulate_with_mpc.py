# simulate_with_mpc.py

import numpy as np
from tube_mpc import TubeMPC
import matplotlib.pyplot as plt
import sys
import os

np.random.seed(42)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.highway_env.vehicle_models import UnicycleModel

model = UnicycleModel(dt=0.12)

# MPC parameters
horizon = 15
u_bounds = (np.array([-2.0, -1.0]), np.array([2.0, 1.0]))
x_bounds = (np.array([-1e6, -1e6, -1e6]), np.array([1e6, 1e6, 1e6]))
disturbance_bound = np.array([0.1, 0.1, 0.05])

# Setup Tube MPC
tube_mpc = TubeMPC(model, horizon, u_bounds, x_bounds, disturbance_bound)

# Initial state
x = np.array([0.0, 0.0, 0.0])
trajectory = [x.copy()]
all_nominal_traj = []
target= np.array([0.5,1.0])
K = np.array([[-0.5, -0.2, 0.0],
              [0.0, 0.0, -0.5]])

# Run for 50 steps
for step in range(50):
    theta_desired = np.arctan2(target[1] - x[1], target[0] - x[0])
    target_traj = np.tile(np.array([target[0], target[1], theta_desired]), (horizon + 1, 1))

    u_nom_seq, x_nom_seq = tube_mpc.plan(x, target_traj)

    if u_nom_seq is None:
        print("MPC failed.")
        break

    # Apply first control with feedback
    x_nominal_current = x_nom_seq[:, 0]
    u_nominal_current = u_nom_seq[:, 0]
    x_error = x - x_nominal_current
    u_corr = u_nominal_current + K @ x_error

    # clip control to original bounds
    u_corr = np.clip(u_corr, tube_mpc.u_bounds[0], tube_mpc.u_bounds[1])

    w = np.random.uniform(-0.1, 0.1)
    x = model.step(x, u_corr+w)
    trajectory.append(x.copy())
    all_nominal_traj.append(x_nom_seq[:, 1])

    # Log Tube MPC and Nominal MPC paths
    print(x,x_nom_seq[:, 1])

    print(
        f"Step {step}: ||pos_error|| = {np.linalg.norm(x[:2] - target[:2]):.3f}, heading_error = {abs(x[2] - theta_desired):.3f}")

    if np.linalg.norm(x[:2] - target[:2]) < 0.02 and abs(x[2] - theta_desired) < 0.1:
        print("Reached target.")
        break

# Plot
trajectory = np.array(trajectory)            # Actual trajectory
nominal_traj = np.array(all_nominal_traj)         # Nominal trajectory: shape (T+1, 3)

start = np.array([0.0, 0.0])
end = np.array([target[0], target[1]])

ax = plt.gca()
ax.set_aspect('equal')

# Plot actual trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], 'blue', label="Actual Trajectory", linewidth=2.5)

# Plot nominal trajectory
if nominal_traj.shape[0] > 0:
    plt.plot(nominal_traj[:, 0], nominal_traj[:, 1], 'orange', label="Nominal Trajectory", linewidth=1)

    # Draw uncertainty tube as a sequence of circles
    tube_radius = 0.05  # Increase this if it's too small to see clearly
    for i in range(nominal_traj.shape[0]):
        circle = plt.Circle((nominal_traj[i, 0], nominal_traj[i, 1]),
                            tube_radius, color='cyan', alpha=0.2, zorder=0)
        ax.add_patch(circle)

# Plot target
plt.plot([start[0], end[0]], [start[1], end[1]], 'r-.', label="Target Line")

plt.grid(True)
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Tube MPC Simulation")
plt.tight_layout()
plt.savefig("tube_mpc_single_vehicle.png")
plt.show()
# plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory")
# plt.plot([start[0], end[0]], [start[1], end[1]], 'r--', label="Target")
# plt.legend()
# plt.grid()
# plt.savefig("nominal_mpc_single_vehicle.png")