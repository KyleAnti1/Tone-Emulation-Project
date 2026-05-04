import numpy as np
import sounddevice as sd
import onnxruntime as ort
import time

# Load ONNX model
sess = ort.InferenceSession("src/deployment/onnx/mlp_overdrive.onnx")

WINDOW_SIZE = 101
SAMPLE_RATE = 44100
buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)

def callback(indata, outdata, frames, time_info, status):
    global buffer

    if status:
        print(f"Status: {status}")

    for i in range(frames):
        # Slide the buffer forward by one sample
        buffer = np.roll(buffer, -1)
        buffer[-1] = indata[i, 0]

        # Run inference
        inp = buffer.reshape(1, WINDOW_SIZE)
        out = sess.run(["output"], {"input": inp})[0]

        outdata[i, 0] = out[0]

# List available devices so you can pick the right one
print(sd.query_devices())

print("\nStarting real-time processing — press Ctrl+C to stop")

with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=64,
        dtype='float32',
        channels=1,
        callback=callback
):
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped")