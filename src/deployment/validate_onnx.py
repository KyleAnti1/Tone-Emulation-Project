import torch
import numpy as np
import onnxruntime as ort
from src.ml.models import MLPModel
import time

SAMPLE_RATE = 44100

# Load PyTorch model
model = MLPModel(input_size=101)
model.load_state_dict(torch.load("models/mlp_baseline.pt"))
model.eval()

# Create test input
test_input = torch.randn(1, 101)

# PyTorch output
with torch.no_grad():
    torch_out = model(test_input).numpy()

# ONNX output
sess = ort.InferenceSession("src/deployment/onnx/mlp_overdrive.onnx")
onnx_out = sess.run(["output"], {"input": test_input.numpy()})[0]

# Compare
diff = np.max(np.abs(torch_out - onnx_out))
print(f"Max difference: {diff:.2e}")

if diff < 1e-5:
    print("Validation passed — ONNX export is correct")
else:
    print("WARNING — outputs differ, check export settings")

# Warm up
for _ in range(10):
    sess.run(["output"], {"input": test_input.numpy()})

# Measure
times = []
for _ in range(1000):
    start = time.perf_counter()
    sess.run(["output"], {"input": test_input.numpy()})
    times.append((time.perf_counter() - start) * 1000)

print(f"Mean inference time: {np.mean(times):.4f} ms")
print(f"Max inference time:  {np.max(times):.4f} ms")
print(f"Real-time safe:      {np.max(times) < (1/SAMPLE_RATE * 1000):.0f}")