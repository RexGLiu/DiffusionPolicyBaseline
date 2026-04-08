import numpy as np

def get_action_space_bounds(action_space):
    low = np.array([box.low for box in action_space])
    high = np.array([box.high for box in action_space])

    return low, high