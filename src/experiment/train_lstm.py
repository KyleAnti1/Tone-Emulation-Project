from src.ml.dataset import AudioPairDataset
from src.ml.models import LSTMModel
from src.ml.train import train, get_loss, get_optimizer
from src.io.audio import load_audio
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader

# Load audio
dry, d_sr = load_audio("data/processed/clean_01_test.wav")
wet, w_sr = load_audio("data/processed/driven_01_test.wav")

# Train/validation split
total_len = len(dry)
split = int(total_len * 0.8)

dry_train, dry_val = dry[:split], dry[split:]
wet_train, wet_val = wet[:split], wet[split:]

# Datasets and loaders
train_dataset = AudioPairDataset(dry_train, wet_train, window_size=101)
val_dataset   = AudioPairDataset(dry_val,   wet_val,   window_size=101)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)

# LSTM model
model     = LSTMModel(input_size=1, hidden_size=32, num_layers=1)
loss_fn   = get_loss()
optimizer = get_optimizer(model)

# Train
train(model, train_loader, val_loader, loss_fn, optimizer, epochs=10)

# Save
torch.save(model.state_dict(), "models/lstm_baseline.pt")

# Evaluate on validation data
model.eval()
x_batch, y_batch = next(iter(val_loader))

with torch.no_grad():
    preds = model(x_batch)

# Waveform plot
plt.figure(figsize=(12, 4))
plt.plot(y_batch[:300].numpy(), label='Target (wet)')
plt.plot(preds[:300].numpy(), label='Prediction', alpha=0.8)
plt.legend()
plt.title('LSTM predictions vs target')
plt.tight_layout()
plt.savefig('results/lstm_prediction_check.png')
plt.show()

# ESR
def esr(target, pred):
    error = target - pred
    return np.mean(error**2) / (np.mean(target**2) + 1e-8)

target_np = y_batch.numpy()
pred_np   = preds.numpy()
print(f"ESR: {esr(target_np, pred_np):.6f}")