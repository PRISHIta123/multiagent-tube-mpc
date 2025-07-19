import numpy as np
import matplotlib.pyplot as plt
from vehicle_models import UnicycleModel

def simulate_single_agent():
    # Initial state: [x, y, theta]
    x0 = np.array([0.0, 0.0, 0.0])

    # Define a control sequence: move forward and turn
    control_sequence = []
    for _ in range(20):
        control_sequence.append([1.0, 0.1])  # move forward with small turn

    # Create model
    model = UnicycleModel(dt=0.1)  # Correct instantiation

    # Simulate
    trajectory = model.simulate_trajectory(x0, control_sequence)

    return trajectory

import numpy as np
import matplotlib.pyplot as plt
from vehicle_models import UnicycleModel

def simulate_multiagent(num_agents=3, timesteps=20):
    # Define shared model (same dynamics for all)
    model = UnicycleModel(dt=0.1)

    # Initial states for each agent (spread out in space)
    initial_states = [
        np.array([i * 2.0, 0.0, np.pi/4 * i]) for i in range(num_agents)
    ]

    # Control sequences for each agent (simple different patterns)
    control_sequences = []
    for i in range(num_agents):
        seq = []
        for t in range(timesteps):
            v = 1.0
            omega = 0.1 * ((-1)**i)  # alternate turning direction
            seq.append([v, omega])
        control_sequences.append(seq)

    # Simulate trajectories for all agents
    trajectories = []
    for i in range(num_agents):
        traj = model.simulate_trajectory(initial_states[i], control_sequences[i])
        trajectories.append(traj)

    return trajectories

if __name__ == "__main__":
    trajectories = simulate_multiagent(num_agents=3, timesteps=25)

    # Plotting
    plt.figure(figsize=(7, 7))
    colors = ['r', 'g', 'b', 'm', 'c']
    for i, traj in enumerate(trajectories):
        x, y, _ = traj.T
        plt.plot(x, y, marker='o', color=colors[i % len(colors)], label=f'Agent {i}')
        plt.plot(x[0], y[0], marker='x', color='k')  # Start point

    plt.title("Multiagent Trajectories - Unicycle Model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("Multiagent_sample_trajectories.png")
    plt.close()


# if __name__ == "__main__":
#     traj = simulate_single_agent()
#     x_vals, y_vals, _ = traj.T
#
#     # Plot trajectory
#     plt.figure(figsize=(6, 6))
#     plt.plot(x_vals, y_vals, marker='o', label="Vehicle path")
#     plt.title("Simulated Trajectory - Unicycle Model")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.grid(True)
#     plt.axis("equal")
#     plt.legend()
#     plt.show()
