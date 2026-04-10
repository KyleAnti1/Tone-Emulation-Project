import numpy as np
import pyaudio
import onnxruntime as ort

# Load ONNX model
sess = ort.InferenceSession("src/deployment/onnx/mlp_overdrive.onnx")

WINDOW_SIZE  = 101
SAMPLE_RATE  = 44100
BUFFER_SIZE  = 64       # smaller = lower latency, harder to run in time
buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)

p = pyaudio.PyAudio()

# Print devices to find your ASIO device
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"{i}: {info['name']} | inputs: {info['maxInputChannels']} | outputs: {info['maxOutputChannels']}")

input_device  = int(input("Enter ASIO input device number: "))
output_device = int(input("Enter ASIO output device number: "))

def callback(in_data, frame_count, time_info, status):
    global buffer

    # Convert raw bytes to float32 array
    indata = np.frombuffer(in_data, dtype=np.float32)

    outdata = np.zeros(frame_count, dtype=np.float32)

    for i in range(frame_count):
        buffer = np.roll(buffer, -1)
        buffer[-1] = indata[i]

        inp = buffer.reshape(1, WINDOW_SIZE)
        out = sess.run(["output"], {"input": inp})[0]
        outdata[i] = out[0]

    return (outdata.tobytes(), pyaudio.paContinue)

# Open stream with ASIO
stream = p.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    output=True,
    input_device_index=input_device,
    output_device_index=output_device,
    frames_per_buffer=BUFFER_SIZE,
    stream_callback=callback
)

print("\nRunning with ASIO — press Ctrl+C to stop")
stream.start_stream()

try:
    import time
    while stream.is_active():
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()