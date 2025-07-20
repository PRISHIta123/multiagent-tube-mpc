import numpy as np
from nominal_mpc import solve_nominal_mpc

class TubeMPC:
    def __init__(self, model, horizon, u_bounds, x_bounds, disturbance_bound):
        self.model = model
        self.horizon = horizon
        self.u_bounds = u_bounds
        self.x_bounds = x_bounds
        self.disturbance_bound = disturbance_bound

    def plan(self, x0, target_traj):
        # Solve the nominal MPC problem with tightened constraints
        u_nom_seq, x_nom_seq = solve_nominal_mpc(
            x0, target_traj, self.x_bounds, self.u_bounds, self.disturbance_bound, self.horizon, self.model.dt
        )

        return u_nom_seq, x_nom_seq