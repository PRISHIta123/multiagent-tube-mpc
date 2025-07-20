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
tube_mpc1 = TubeMPC(model, horizon, u_bounds, x_bounds, disturbance_bound)
tube_mpc2 = TubeMPC(model, horizon, u_bounds, x_bounds, disturbance_bound)

# Initial state
K1 = np.array([[-0.5, -0.2, 0.0],
              [0.0, 0.0, -0.5]])
K2 = np.array([[-0.5, -0.2, 0.0],
              [0.0, 0.0, -0.5]])

x1 = np.array([0.0, 0.0, 0.0])
x2 = np.array([0.0, 1.2, 0.0])

trajectory1 = [x1.copy()]
trajectory2 = [x2.copy()]

all_nominal_traj1 = []
all_nominal_traj2 = []

target1= np.array([0.5,1.0])
target2= np.array([1.0,0.75])

# Run for 50 steps
for step in range(50):

    theta_desired1 = np.arctan2(target1[1] - x1[1], target1[0] - x1[0])
    theta_desired2 = np.arctan2(target2[1] - x2[1], target2[0] - x2[0])

    target_traj1 = np.tile(np.array([target1[0], target1[1], theta_desired1]), (horizon + 1, 1))
    target_traj2 = np.tile(np.array([target2[0], target2[1], theta_desired2]), (horizon + 1, 1))

    u_nom_seq1, x_nom_seq1 = tube_mpc1.plan(x1, target_traj1)
    u_nom_seq2, x_nom_seq2 = tube_mpc2.plan(x2, target_traj2)

    if u_nom_seq1 is None:
        print("MPC 1 failed.")
        break

    if u_nom_seq2 is None:
        print("MPC 2 failed.")
        break

    # Apply first control with feedback
    x_nominal_current1 = x_nom_seq1[:, 0]
    u_nominal_current1 = u_nom_seq1[:, 0]
    x_error1 = x1 - x_nominal_current1
    u_corr1 = u_nominal_current1 + K1 @ x_error1

    x_nominal_current2 = x_nom_seq2[:, 0]
    u_nominal_current2 = u_nom_seq2[:, 0]
    x_error2 = x2 - x_nominal_current2
    u_corr2 = u_nominal_current2 + K2 @ x_error2

    # clip control to original bounds
    u_corr1 = np.clip(u_corr1, tube_mpc1.u_bounds[0], tube_mpc1.u_bounds[1])
    u_corr2 = np.clip(u_corr2, tube_mpc2.u_bounds[0], tube_mpc2.u_bounds[1])

    w = np.random.uniform(-0.1, 0.1)

    x1 = model.step(x1, u_corr1+w)
    trajectory1.append(x1.copy())
    all_nominal_traj1.append(x_nom_seq1[:, 1])

    x2 = model.step(x2, u_corr2 + w)
    trajectory2.append(x2.copy())
    all_nominal_traj2.append(x_nom_seq2[:, 1])

    # Log Tube MPC and Nominal MPC paths
    print("Tube MPC1:",x1,x_nom_seq1[:, 1])

    print(
        f"Step {step}: ||pos_error|| = {np.linalg.norm(x1[:2] - target1[:2]):.3f}, heading_error = {abs(x1[2] - theta_desired1):.3f}")

    print("Tube MPC2:", x2, x_nom_seq2[:, 1])

    print(
        f"Step {step}: ||pos_error|| = {np.linalg.norm(x2[:2] - target2[:2]):.3f}, heading_error = {abs(x2[2] - theta_desired2):.3f}")

    flag1=0
    flag2=0

    if np.linalg.norm(x1[:2] - target1[:2]) < 0.02 and abs(x1[2] - theta_desired1) < 0.1:
        print("Reached target 1.")
        flag1=1

    if np.linalg.norm(x2[:2] - target2[:2]) < 0.02 and abs(x2[2] - theta_desired2) < 0.1:
        print("Reached target 2.")
        flag2=1

    #If both agents reached their goals end loop
    if flag1==1 and flag2==1:
        break

trajectory1 = np.array(trajectory1)
trajectory2 = np.array(trajectory2)
all_nominal_traj1 = np.array(all_nominal_traj1)
all_nominal_traj2 = np.array(all_nominal_traj2)

# Plotting
plt.figure(figsize=(8,8)); ax = plt.gca(); ax.set_aspect('equal')

# Trajectories
plt.plot(trajectory1[:,0], trajectory1[:,1], 'blue', label='Agent 1 – Actual', linewidth=2.5)
plt.plot(all_nominal_traj1[:,0], all_nominal_traj1[:,1], 'r', label='Agent 1 – Nominal', linewidth=1)
plt.plot(trajectory2[:,0], trajectory2[:,1], 'red', label='Agent 2 – Actual', linewidth=2.5)
plt.plot(all_nominal_traj2[:,0], all_nominal_traj2[:,1], 'g', label='Agent 2 – Nominal', linewidth=1)

start1 = np.array([0.0, 0.0])
end1 = np.array([target1[0], target1[1]])
plt.plot([start1[0], end1[0]], [start1[1], end1[1]], 'black', linestyle='--',label="Target Line 1")

start2 = np.array([0.0, 1.2])
end2 = np.array([target2[0], target2[1]])
plt.plot([start2[0], end2[0]], [start2[1], end2[1]], 'black', linestyle='--', label="Target Line 2")

# Tubes
for (x, y, _) in all_nominal_traj1:
    circ = plt.Circle((x, y), 0.12, color='cyan', alpha=0.1)
    ax.add_patch(circ)
for (x, y, _) in all_nominal_traj2:
    circ = plt.Circle((x, y), 0.12, color='salmon', alpha=0.1)
    ax.add_patch(circ)

# Start/goal markers
plt.scatter(*trajectory1[0,:2], c='black', marker='o')
plt.scatter(*target1, c='black', marker='X', s=50)
plt.scatter(*trajectory2[0,:2], c='black', marker='o')
plt.scatter(*target2, c='black', marker='X', s=50)

plt.grid()
plt.title("2-Agent Tube MPC with Disturbances")
plt.legend()
plt.savefig("tube_mpc_multi_vehicle.png")
plt.show()

#
#
# # Plot
# trajectory = np.array(trajectory)            # Actual trajectory
# nominal_traj = np.array(all_nominal_traj)         # Nominal trajectory: shape (T+1, 3)
#
# start = np.array([0.0, 0.0])
# end = np.array([target[0], target[1]])
#
# ax = plt.gca()
# ax.set_aspect('equal')
#
# # Plot actual trajectory
# plt.plot(trajectory[:, 0], trajectory[:, 1], 'blue', label="Actual Trajectory", linewidth=2.5)
#
# # Plot nominal trajectory
# if nominal_traj.shape[0] > 0:
#     plt.plot(nominal_traj[:, 0], nominal_traj[:, 1], 'orange', label="Nominal Trajectory", linewidth=1)
#
#     # Draw uncertainty tube as a sequence of circles
#     tube_radius = 0.05  # Increase this if it's too small to see clearly
#     for i in range(nominal_traj.shape[0]):
#         circle = plt.Circle((nominal_traj[i, 0], nominal_traj[i, 1]),
#                             tube_radius, color='cyan', alpha=0.2, zorder=0)
#         ax.add_patch(circle)
#
# # Plot target
# plt.plot([start[0], end[0]], [start[1], end[1]], 'r-.', label="Target Line")
#
# plt.grid(True)
# plt.legend()
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Tube MPC Simulation")
# plt.tight_layout()
# plt.savefig("tube_mpc_single_vehicle.png")
# plt.show()
# plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory")
# plt.plot([start[0], end[0]], [start[1], end[1]], 'r--', label="Target")
# plt.legend()
# plt.grid()
# plt.savefig("nominal_mpc_single_vehicle.png")