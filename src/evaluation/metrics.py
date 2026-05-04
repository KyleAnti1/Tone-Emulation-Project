import numpy as np
from scipy import signal as scipy_signal


def esr(target, pred):
    """Error-to-Signal Ratio. Lower is better. <0.01 is good."""
    error = target - pred
    return np.mean(error**2) / (np.mean(target**2) + 1e-8)


def rmse(target, pred):
    """Root Mean Square Error. Lower is better."""
    return np.sqrt(np.mean((target - pred)**2))


def mae(target, pred):
    """Mean Absolute Error. Lower is better."""
    return np.mean(np.abs(target - pred))


def psnr(target, pred):
    """
    Peak Signal-to-Noise Ratio in dB. Higher is better.
    Uses signal peak rather than fixed maximum.
    """
    mse = np.mean((target - pred)**2)
    if mse == 0:
        return float('inf')
    max_val = np.max(np.abs(target))
    return 20 * np.log10(max_val / (np.sqrt(mse) + 1e-8))


def pearson_correlation(target, pred):
    """
    Pearson correlation coefficient. Closer to 1.0 is better.
    Measures shape similarity independent of amplitude.
    """
    target_mean = target - np.mean(target)
    pred_mean   = pred   - np.mean(pred)
    numerator   = np.sum(target_mean * pred_mean)
    denominator = np.sqrt(np.sum(target_mean**2) * np.sum(pred_mean**2))
    return numerator / (denominator + 1e-8)


def spectral_convergence(target, pred, sr, n_fft=2048):
    """
    Spectral Convergence. Lower is better.
    Measures how well frequency content is reproduced.
    """
    _, _, target_stft = scipy_signal.stft(target, fs=sr, nperseg=n_fft)
    _, _, pred_stft   = scipy_signal.stft(pred,   fs=sr, nperseg=n_fft)

    target_mag = np.abs(target_stft)
    pred_mag   = np.abs(pred_stft)

    numerator   = np.linalg.norm(target_mag - pred_mag, 'fro')
    denominator = np.linalg.norm(target_mag, 'fro')
    return numerator / (denominator + 1e-8)


def evaluate_all(target, pred, sr, model_name="Model"):
    """Run all metrics and print a formatted report."""
    metrics = {
        "ESR":                  esr(target, pred),
        "RMSE":                 rmse(target, pred),
        "MAE":                  mae(target, pred),
        "PSNR (dB)":            psnr(target, pred),
        "Pearson r":            pearson_correlation(target, pred),
        "Spectral Convergence": spectral_convergence(target, pred, sr),
    }

    print(f"\n{'='*45}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'='*45}")
    print(f"  {'Metric':<25} {'Value'}")
    print(f"  {'-'*38}")
    for name, value in metrics.items():
        print(f"  {name:<25} {value:.6f}")
    print(f"{'='*45}\n")

    return metrics