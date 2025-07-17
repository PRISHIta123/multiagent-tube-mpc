import numpy as np
from nominal_mpc import solve_nominal_mpc
from constraints import tighten_constraints
class TubeMPC:
    def __init__(self, model, horizon, Q, R, Qf, u_bounds, x_bounds, disturbance_bound):
        self.model = model
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.u_bounds = u_bounds
        self.x_bounds = x_bounds
        self.disturbance_bound = disturbance_bound

    def plan(self, x0, target_traj):
        # Tighten constraints to account for disturbances
        tightened_x_bounds = tighten_constraints(self.x_bounds, self.disturbance_bound)

        # Solve the nominal MPC problem with tightened constraints
        u_nom_seq, x_nom_seq = solve_nominal_mpc(
            self.model, x0, target_traj, self.horizon,
            self.Q, self.R, self.Qf, self.u_bounds, tightened_x_bounds
        )
        return u_nom_seq, x_nom_seq