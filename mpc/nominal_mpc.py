import numpy as np
import cvxpy as cp

def solve_nominal_mpc(model, x0, target_traj, horizon, Q, R, Qf, u_bounds, x_bounds):
    nx = x0.shape[0]
    nu = u_bounds.shape[1]

    x = cp.Variable((horizon + 1, nx))
    u = cp.Variable((horizon, nu))

    cost = 0
    constraints = [x[0] == x0]

    for t in range(horizon):
        cost += cp.quad_form(x[t] - target_traj[t], Q) + cp.quad_form(u[t], R)
        constraints += [x[t + 1] == model(x[t], u[t])]
        constraints += [x_bounds[0] <= x[t], x[t] <= x_bounds[1]]
        constraints += [u_bounds[0] <= u[t], u[t] <= u_bounds[1]]

    cost += cp.quad_form(x[horizon] - target_traj[horizon], Qf)
    constraints += [x_bounds[0] <= x[horizon], x[horizon] <= x_bounds[1]]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)

    return u.value, x.value