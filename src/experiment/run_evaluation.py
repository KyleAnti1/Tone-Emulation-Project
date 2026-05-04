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
dry, sr = load_audio("data/processed/training_data_10_clean.wav")
wet, _  = load_audio("data/processed/training_data_10_driven.wav")

# --- Val split ---
total_len = len(dry)
split     = int(total_len * 0.8)

dry_val = dry[split:]
wet_val = wet[split:]

val_dataset = AudioPairDataset(dry_val, wet_val, window_size=101)
val_loader  = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# Load models
mlp = MLPModel(input_size=101)
mlp.load_state_dict(torch.load("models/new_mlp.pt"))
mlp.eval()

lstm = LSTMModel(input_size=1, hidden_size=32, num_layers=1)
lstm.load_state_dict(torch.load("models/lstm_baseline.pt"))
lstm.eval()

# ESR function
def esr(target, pred):
    error = target - pred
    return np.mean(error**2) / (np.mean(target**2) + 1e-8)

# Get predictions for a model over full val set
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

print(len(y_batch))
plt.figure(figsize=(12, 4))
plt.plot(y_batch[:1024].numpy(),  label='Target (wet)', linewidth=2)
plt.plot(mlp_batch_preds[:1024],  label='MLP',  alpha=0.8)
plt.plot(lstm_batch_preds[:1024], label='LSTM', alpha=0.8)
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

from src.evaluation.metrics import evaluate_all

# Evaluate all metrics on full validation set
mlp_metrics  = evaluate_all(mlp_targets,  mlp_preds,  sr, model_name="MLP")
lstm_metrics = evaluate_all(lstm_targets, lstm_preds, sr, model_name="LSTM")

def print_comparison_table(mlp_metrics, lstm_metrics):
    print(f"\n{'='*60}")
    print(f"  {'Metric':<25} {'MLP':<15} {'LSTM':<15} {'Better'}")
    print(f"  {'-'*55}")

    lower_is_better = ["ESR", "RMSE", "MAE", "Spectral Convergence"]

    for metric in mlp_metrics:
        mlp_val  = mlp_metrics[metric]
        lstm_val = lstm_metrics[metric]

        if metric in lower_is_better:
            winner = "MLP" if mlp_val < lstm_val else "LSTM"
        else:
            winner = "MLP" if mlp_val > lstm_val else "LSTM"

        print(f"  {metric:<25} {mlp_val:<15.6f} {lstm_val:<15.6f} {winner}")

    print(f"{'='*60}\n")

print_comparison_table(mlp_metrics, lstm_metrics)

print("\nAll evaluation complete. Figures saved to results/")