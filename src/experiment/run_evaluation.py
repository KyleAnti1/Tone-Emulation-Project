import torch
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import time
from torch.utils.data import DataLoader

from src.ml.models import MLPModel, LSTMModel
from src.ml.dataset import AudioPairDataset
from src.io.audio import load_audio
from src.evaluation.spectral import plot_spectral_comparison

# --- Load audio ---
dry, sr = load_audio("data/processed/clean_01_test.wav")
wet, _  = load_audio("data/processed/driven_01_test.wav")

# --- Val split ---
total_len = len(dry)
split     = int(total_len * 0.8)

dry_val = dry[split:]
wet_val = wet[split:]

val_dataset = AudioPairDataset(dry_val, wet_val, window_size=101)
val_loader  = DataLoader(val_dataset, batch_size=512, shuffle=False)

# --- Load models ---
mlp = MLPModel(input_size=101)
mlp.load_state_dict(torch.load("models/mlp_baseline.pt"))
mlp.eval()

lstm = LSTMModel(input_size=1, hidden_size=32, num_layers=1)
lstm.load_state_dict(torch.load("models/lstm_baseline.pt"))
lstm.eval()

# --- ESR function ---
def esr(target, pred):
    error = target - pred
    return np.mean(error**2) / (np.mean(target**2) + 1e-8)

# --- Get predictions for a model over full val set ---
def get_all_predictions(model, loader):
    all_preds   = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            all_preds.append(preds.numpy())
            all_targets.append(y.numpy())
    return np.concatenate(all_targets), np.concatenate(all_preds)

# --- Evaluate both models ---
print("Evaluating MLP...")
mlp_targets, mlp_preds = get_all_predictions(mlp, val_loader)
mlp_esr = esr(mlp_targets, mlp_preds)

print("Evaluating LSTM...")
lstm_targets, lstm_preds = get_all_predictions(lstm, val_loader)
lstm_esr = esr(lstm_targets, lstm_preds)

# --- Count parameters ---
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

mlp_params  = count_params(mlp)
lstm_params = count_params(lstm)

# --- Inference latency (ONNX) ---
sess       = ort.InferenceSession("src/deployment/onnx/mlp_overdrive.onnx")
test_input = np.random.randn(1, 101).astype(np.float32)

for _ in range(10):
    sess.run(["output"], {"input": test_input})

times = []
for _ in range(1000):
    start = time.perf_counter()
    sess.run(["output"], {"input": test_input})
    times.append((time.perf_counter() - start) * 1000)

mean_inf = np.mean(times)
max_inf  = np.max(times)
budget   = (1 / sr) * 1000

# --- Summary table ---
print("\n" + "=" * 65)
print(f"{'Model':<10} {'ESR':<12} {'Params':<10} {'Mean Inf (ms)':<16} {'RT Safe'}")
print("-" * 65)
print(f"{'MLP':<10} {mlp_esr:<12.6f} {mlp_params:<10} {mean_inf:<16.4f} {str(max_inf < budget)}")
print(f"{'LSTM':<10} {lstm_esr:<12.6f} {lstm_params:<10} {'N/A (CPU)':<16} {'N/A'}")
print("=" * 65)
print(f"\nSample budget at {sr}Hz : {budget:.4f} ms")
print(f"MLP max inference time : {max_inf:.4f} ms")

# --- Waveform comparison plot (MLP vs LSTM vs target) ---
x_batch, y_batch = next(iter(val_loader))

with torch.no_grad():
    mlp_batch_preds  = mlp(x_batch).numpy()
    lstm_batch_preds = lstm(x_batch).numpy()

plt.figure(figsize=(12, 4))
plt.plot(y_batch[:300].numpy(),  label='Target (wet)', linewidth=2)
plt.plot(mlp_batch_preds[:300],  label='MLP',  alpha=0.8)
plt.plot(lstm_batch_preds[:300], label='LSTM', alpha=0.8)
plt.legend()
plt.title('MLP vs LSTM — Predictions vs Target')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig('results/model_comparison_waveform.png', dpi=150)
plt.show()

# --- Error distribution comparison ---
mlp_error  = mlp_targets  - mlp_preds
lstm_error = lstm_targets - lstm_preds

plt.figure(figsize=(12, 4))
plt.hist(mlp_error,  bins=100, alpha=0.6, label='MLP',  edgecolor='none')
plt.hist(lstm_error, bins=100, alpha=0.6, label='LSTM', edgecolor='none')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Error')
plt.ylabel('Count')
plt.title('Error Distribution — MLP vs LSTM')
plt.legend()
plt.tight_layout()
plt.savefig('results/error_distribution_comparison.png', dpi=150)
plt.show()

# Spectral comparison
plot_spectral_comparison(
    dry, wet, sr,
    models_dict={"MLP": mlp, "LSTM": lstm}
)

print("\nAll evaluation complete. Figures saved to results/")