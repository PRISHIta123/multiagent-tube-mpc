import casadi as ca
import numpy as np
from constraints import tighten_constraints

def solve_nominal_mpc(x0, target_traj, x_bounds, u_bounds, disturbance_bound, T, dt=0.12):

    target_traj = target_traj.T

    nx = 3  # state dimension
    nu = 2  # control dimension

    # MPC weights
    Q = ca.diagcat(10, 10, 1)  # position, position, heading
    R = ca.diagcat(1, 1)       # control effort

    # Compute tube radius (disturbance bound as tube radius)
    tube_radius = np.array(disturbance_bound)

    # Tighten x_bounds and u_bounds
    tightened_x_bounds = tighten_constraints(x_bounds, disturbance_bound)

    # Conservative control tightening (say 10% of input range or based on LQR)
    #u_margin = 0.1 * (u_bounds[1] - u_bounds[0])
    #tightened_u_bounds = tighten_constraints(u_bounds,u_margin)
    tightened_u_bounds= u_bounds

    # Optimization variables
    opti = ca.Opti()
    X = opti.variable(nx, T + 1)  # state trajectory
    U = opti.variable(nu, T)      # control trajectory

    # Parameters (initial condition and reference)
    x_init = opti.parameter(nx)
    ref_traj = opti.parameter(nx, T)

    # Initial condition constraint
    opti.subject_to(X[:, 0] == x_init)

    # System dynamics (unicycle)
    def dynamics(x, u):
        return ca.vertcat(
            u[0] * ca.cos(x[2]),
            u[0] * ca.sin(x[2]),
            u[1]
        )

    cost = 0
    for t in range(T):
        # Dynamics constraint
        x_next = X[:, t] + dt * dynamics(X[:, t], U[:, t])
        opti.subject_to(X[:, t + 1] == x_next)

        # State constraints (tightened tube)
        opti.subject_to(tightened_x_bounds[0] <= X[:, t])
        opti.subject_to(X[:, t] <= tightened_x_bounds[1])

        # Input constraints (tightened tube)
        opti.subject_to(tightened_u_bounds[0] <= U[:, t])
        opti.subject_to(U[:, t] <= tightened_u_bounds[1])

        # Cost
        x_err = ca.reshape(X[:, t] - ref_traj[:, t], (nx, 1))
        u_t = ca.reshape(U[:, t], (nu, 1))
        cost += x_err.T @ Q @ x_err + u_t.T @ R @ u_t

    # Terminal state constraint
    opti.subject_to(tightened_x_bounds[0] <= X[:, T])
    opti.subject_to(X[:, T] <= tightened_x_bounds[1])

    # Terminal cost
    x_err_terminal = ca.reshape(X[:, T - 1] - ref_traj[:, T - 1], (nx, 1))
    cost += x_err_terminal.T @ Q @ x_err_terminal

    opti.minimize(cost)

    # Solver setup
    opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt", opts)

    # Set parameter values
    opti.set_value(x_init, x0)
    opti.set_value(ref_traj, target_traj[:, :T])

    # Solve
    sol = opti.solve()

    # Extract solution
    u_nom = sol.value(U)
    x_nom = sol.value(X)
    return u_nom, x_nom
