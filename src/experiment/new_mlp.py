from src.ml.dataset import AudioPairDataset
from src.ml.models import MLPModel
from src.ml.train import train, get_loss, get_optimizer
from src.io.audio import load_audio
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader

# Load audio
dry_train, dt_sr = load_audio("data/processed/training_data_10_clean.wav")
wet_train, wt_sr = load_audio("data/processed/training_data_10_driven.wav")
# dry_val, dv_sr = load_audio("data/processed/validation_data_10_clean.wav")
# wet_val, wv_sr = load_audio("data/processed/validation_data_10_driven.wav")

"""
fig, axes = plt.subplots(2, 2, figsize=(14, 6))

axes[0,0].plot(dry_train[:5000])
axes[0,0].set_title('Train dry (first 5000 samples)')

axes[0,1].plot(wet_train[:5000])
axes[0,1].set_title('Train wet (first 5000 samples)')

axes[1,0].plot(dry_val[:5000])
axes[1,0].set_title('Val dry (first 5000 samples)')

axes[1,1].plot(wet_val[:5000])
axes[1,1].set_title('Val wet (first 5000 samples)')

plt.tight_layout()
plt.savefig('results/signal_diagnostic.png', dpi=150)
plt.show()
"""

# Train/validation split
total_len = len(dry_train)
split = int(total_len * 0.8)

dry_train, dry_val = dry_train[:split], dry_train[split:]
wet_train, wet_val = wet_train[:split], wet_train[split:]

# Datasets and loaders
train_dataset = AudioPairDataset(dry_train, wet_train, window_size=101)
val_dataset   = AudioPairDataset(dry_val,   wet_val,   window_size=101)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)

# Model, loss, optimiser
model     = MLPModel(input_size=101)
loss_fn   = get_loss()
optimizer = get_optimizer(model)

# Train
train(model, train_loader, val_loader, loss_fn, optimizer, epochs=10)

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
plt.title('MLP predictions vs target')
plt.tight_layout()
plt.savefig('results/prediction_check.png')
plt.show()

# ESR
def esr(target, pred):
    error = target - pred
    return np.mean(error**2) / (np.mean(target**2) + 1e-8)

target_np = y_batch.numpy()
pred_np   = preds.numpy()
print(f"ESR: {esr(target_np, pred_np):.6f}")

torch.save(model.state_dict(), "models/new_mlp.pt")