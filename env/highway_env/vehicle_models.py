from casadi import SX, vertcat, Function
import numpy as np

class UnicycleModel:
    def __init__(self, dt=0.1, wheelbase=0.5):
        self.dt = dt
        self.L = wheelbase  # Wheelbase length

        # Define symbolic dynamics for CasADi
        x = SX.sym('x', 3)  # [x, y, theta]
        u = SX.sym('u', 2)  # [v, w]

        x_next = vertcat(
            x[0] + u[0] * np.cos(x[2]) * self.dt,
            x[1] + u[0] * np.sin(x[2]) * self.dt,
            x[2] + u[1] * self.dt
        )
        self.f_sym = Function('f', [x, u], [x_next])

    def step(self, state, control):
        """
        Propagate the state using control inputs.
        state: [x, y, theta]
        control: [v, omega] (linear and angular velocities)
        """
        x, y, theta = state
        v, omega = control

        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt

        return np.array([x_next, y_next, theta_next])

    def step_symbolic(self, x, u):
        return self.f_sym(x, u)

    def simulate_trajectory(self, initial_state, control_sequence):
        """
        Simulates a trajectory given a control sequence.
        control_sequence: List of [v, omega] at each timestep.
        """
        trajectory = [initial_state]
        state = initial_state

        for control in control_sequence:
            state = self.step(state, control)
            trajectory.append(state)

        return np.array(trajectory)
