import time
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("src/deployment/onnx/mlp_overdrive.onnx")
test_input = np.random.randn(1, 101).astype(np.float32)
SAMPLE_RATE = 48000

# Warm up
for _ in range(10):
    sess.run(["output"], {"input": test_input})

# Measure per-sample inference
times = []
for _ in range(1000):
    start = time.perf_counter()
    sess.run(["output"], {"input": test_input})
    times.append((time.perf_counter() - start) * 1000)

mean_inf_ms  = np.mean(times)
max_inf_ms   = np.max(times)
std_inf_ms   = np.std(times)
budget_ms    = (1 / SAMPLE_RATE) * 1000

# Real-time factor
# Time to process 1 second of audio = mean_inf_ms * SAMPLE_RATE
time_to_process_1s = (mean_inf_ms / 1000) * SAMPLE_RATE
rtf = time_to_process_1s / 1.0

# Buffer latency
buffer_size    = 256
buffer_lat_ms  = (buffer_size / SAMPLE_RATE) * 1000
total_lat_ms   = (2 * buffer_lat_ms) + mean_inf_ms

print(f"\n{'='*50}")
print(f"  Inference Latency Report")
print(f"{'='*50}")
print(f"  Mean inference time : {mean_inf_ms:.4f} ms")
print(f"  Max inference time  : {max_inf_ms:.4f} ms")
print(f"  Std inference time  : {std_inf_ms:.4f} ms")
print(f"  Per-sample budget   : {budget_ms:.4f} ms")
print(f"  Real-time factor    : {rtf:.6f}")
print(f"  RT capable          : {max_inf_ms < budget_ms}")
print(f"  Buffer latency      : {buffer_lat_ms:.2f} ms")
print(f"  Est. total latency  : {total_lat_ms:.2f} ms")
print(f"{'='*50}")