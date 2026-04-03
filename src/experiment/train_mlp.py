from src.ml.dataset import AudioPairDataset
from src.ml.models import MLPModel
from src.ml.train import train, get_loss, get_optimizer
from src.io.audio import load_audio
import matplotlib.pyplot as plt
import torch
import numpy as np

from torch.utils.data import DataLoader

dry, d_sr = load_audio("data/processed/clean_01_test.wav")
wet, w_sr = load_audio("data/processed/driven_01_test.wav")

dataset = AudioPairDataset(dry, wet, window_size=101)

loader = DataLoader(dataset, batch_size=512, shuffle=True)

model = MLPModel(input_size=101)

loss_fn = get_loss()
optimizer = get_optimizer(model)

train(model, loader, loss_fn, optimizer, epochs=10)

model.eval()
x_batch, y_batch = next(iter(loader))

with torch.no_grad():
    preds = model(x_batch)


'''
plt.figure(figsize=(12, 4))
plt.plot(y_batch[:300].numpy(), label='Target (wet)')
plt.plot(preds[:300].numpy(), label='Prediction', alpha=0.8)
plt.legend()
plt.title('MLP predictions vs target')
plt.tight_layout()
plt.savefig('results/prediction_check.png')
plt.show()
'''

def esr(target, pred):
    error = target - pred
    return np.mean(error**2) / (np.mean(target**2) + 1e-8)

target_np = y_batch.numpy()
pred_np = preds.numpy()
print(f"ESR: {esr(target_np, pred_np):.6f}")
