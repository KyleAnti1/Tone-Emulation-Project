from scipy.signal import correlate
import numpy as np

def align_signals(dry, wet):
    # function to align wet and dry signals
    corr = correlate(wet, dry, mode='full')
    lag = np.argmax(corr) - len(dry) + 1

    if lag > 0:
        wet_aligned = wet[lag:]
        dry_aligned = dry[:len(wet_aligned)]
    else:
        dry_aligned = dry[-lag:]
        wet_aligned = wet[:len(dry_aligned)]

    min_len = min(len(dry_aligned), len(wet_aligned))
    return dry_aligned[:min_len], wet_aligned[:min_len]

