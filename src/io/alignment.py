from scipy.signal import correlate, hilbert
import numpy as np

def align_signals(dry, wet):
    # function to align wet and dry signals
    # Use signal envelopes for robust alignment
    dry_env = np.abs(hilbert(dry))
    wet_env = np.abs(hilbert(wet))

    corr = correlate(wet_env, dry_env, mode='full')
    lag = np.argmax(corr) - len(dry) + 1

    print(f"Detected lag: {lag} samples")

    if lag > 0:
        wet_aligned = wet[lag:]
        dry_aligned = dry[:len(wet_aligned)]
    else:
        dry_aligned = dry[-lag:]
        wet_aligned = wet[:len(dry_aligned)]

    min_len = min(len(dry_aligned), len(wet_aligned))
    return dry_aligned[:min_len], wet_aligned[:min_len]

