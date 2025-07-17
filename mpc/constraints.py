import numpy as np

def tighten_constraints(x_bounds, disturbance_bound):
    lower_bound, upper_bound = x_bounds
    tightened_lower = lower_bound + disturbance_bound
    tightened_upper = upper_bound - disturbance_bound
    return (tightened_lower, tightened_upper)