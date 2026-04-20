import numpy as np

def remove_dc(x):
    return x - np.mean(x)

def normalise(x, target_peak=0.9):
    peak = np.max(np.abs(x))
    if peak > 0:
        return x * (target_peak / peak)

    elif peak  == 0:
        print ("Warning: silent file")
    return x