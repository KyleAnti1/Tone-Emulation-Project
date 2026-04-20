import numpy as np
import matplotlib.pyplot as plt
import torch
import soundfile as sf
from src.ml.models import MLPModel, LSTMModel
from src.ml.dataset import AudioPairDataset
from src.io.audio import load_audio
from torch.utils.data import DataLoader

def compute_spectrum(signal, sr):
    # Compute magnitude spectrum in dB
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1/sr)
    magnitude = np.abs(np.fft.rfft(signal))
    magnitude_db = 20 * np.log10(magnitude + 1e-8)
    return freqs, magnitude_db

def smooth_spectrum(magnitude_db, window=50):
    # Smooth spectrum for cleaner plot
    return np.convolve(magnitude_db,
                       np.ones(window)/window,
                       mode='same')

def get_predictions(model, dry, wet, window_size=101):
    # Run model over full validation signal
    total_len = len(dry)
    split = int(total_len * 0.8)

    dry_val = dry[split:]
    wet_val = wet[split:]

    dataset = AudioPairDataset(dry_val, wet_val, window_size=window_size)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            all_preds.append(preds.numpy())

    return np.concatenate(all_preds), wet_val[window_size//2 : window_size//2 + len(np.concatenate(all_preds))]

def plot_spectral_comparison(dry, wet, sr, models_dict, window_size=101):
    """
    models_dict: {"MLP": mlp_model, "LSTM": lstm_model}
    """

    total_len = len(dry)
    split = int(total_len * 0.8)
    dry_val = dry[split:]
    wet_val = wet[split:]

    # Compute target spectrum
    freqs, wet_db = compute_spectrum(wet_val, sr)
    wet_db_smooth = smooth_spectrum(wet_db)

    # Compute dry spectrum for reference
    _, dry_db = compute_spectrum(dry_val, sr)
    dry_db_smooth = smooth_spectrum(dry_db)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.semilogx(freqs, dry_db_smooth,
                 label='Dry (input)',
                 color='gray',
                 linestyle='--',
                 alpha=0.7)
    plt.semilogx(freqs, wet_db_smooth,
                 label='Target (wet)',
                 color='blue',
                 linewidth=2)

    # Plot each model's prediction
    colors = ['orange', 'green', 'red']
    for (name, model), color in zip(models_dict.items(), colors):
        preds, _ = get_predictions(model, dry, wet, window_size)
        freqs_p, pred_db = compute_spectrum(preds, sr)
        pred_db_smooth = smooth_spectrum(pred_db)
        plt.semilogx(freqs_p, pred_db_smooth,
                     label=f'{name} prediction',
                     color=color,
                     alpha=0.8)

    # Mark the Tube Screamer frequency regions
    plt.axvspan(700, 1000, alpha=0.08, color='yellow',
                label='Tube Screamer mid boost region')
    plt.axvline(x=150,  color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=2000, color='gray', linestyle=':', alpha=0.5)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Spectral Comparison — Dry vs Target vs Model Predictions')
    plt.legend()
    plt.xlim([20, sr/2])
    plt.ylim([-100, None])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/spectral_comparison.png', dpi=150)
    plt.show()