import pyaudio
import numpy as np

p = pyaudio.PyAudio()

print("=== Available Input Devices ===")
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    if d['maxInputChannels'] > 0:
        print(f"{i}: {d['name']} | in: {d['maxInputChannels']}")

print("\n=== Testing M-Audio Input (device 1) at 44100Hz ===")
print("Play your guitar now — listening for 5 seconds...\n")

try:
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=2,
        rate=44100,
        input=True,
        input_device_index=1,
        frames_per_buffer=1024
    )

    signal_detected = False
    for _ in range(5 * 44100 // 1024):
        data = stream.read(1024, exception_on_overflow=False)
        arr  = np.frombuffer(data, dtype=np.float32)
        rms  = np.sqrt(np.mean(arr**2))
        if rms > 0.001:
            print(f"SIGNAL DETECTED: RMS={rms:.4f}")
            signal_detected = True

    stream.stop_stream()
    stream.close()

    if not signal_detected:
        print("No signal detected at 44100Hz — trying 48000Hz...")

        stream2 = p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=48000,
            input=True,
            input_device_index=1,
            frames_per_buffer=1024
        )

        print("Listening at 48000Hz for 5 seconds...\n")
        for _ in range(5 * 48000 // 1024):
            data = stream2.read(1024, exception_on_overflow=False)
            arr  = np.frombuffer(data, dtype=np.float32)
            rms  = np.sqrt(np.mean(arr**2))
            if rms > 0.001:
                print(f"SIGNAL DETECTED at 48000Hz: RMS={rms:.4f}")

        stream2.stop_stream()
        stream2.close()

except Exception as e:
    print(f"Error opening stream: {e}")

p.terminate()
print("\nDone.")