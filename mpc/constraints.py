import numpy as np

def tighten_constraints(bounds, tightening_radius):
    lower, upper = bounds
    tightened_lower = lower + tightening_radius
    tightened_upper = upper - tightening_radius
    return tightened_lower, tightened_upper