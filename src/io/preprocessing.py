import numpy as np

def remove_dc(x):
    return x - np.mean(x)