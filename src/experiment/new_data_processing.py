from src.io.alignment import align_signals
from src.io.audio import load_audio
from src.io.preprocessing import *
import soundfile as sf
import os
import matplotlib.pyplot as plt

dry_train, dt_sr = load_audio("data/raw/training_data_10_clean.wav")
wet_train, wt_sr = load_audio("data/raw/training_data_10_driven.wav")
dry_val, dv_sr = load_audio("data/raw/validation_data_10_clean.wav")
wet_val, wv_sr = load_audio("data/raw/validation_data_10_driven.wav")

assert dt_sr == wt_sr, f"Sample rate mismatch: {dt_sr} vs {wt_sr}"

dc_dry_test = remove_dc(dry_train)
dc_wet_test = remove_dc(wet_train)
dc_dry_val = remove_dc(dry_val)
dc_wet_val = remove_dc(wet_val)

norm_dt = normalise(dc_dry_test)
norm_wt = normalise(dc_wet_test)
norm_dv = normalise(dc_dry_val)
norm_wv = normalise(dc_wet_val)

processed_dt, processed_wt = align_signals(norm_dt, norm_wt)
processed_dv, processed_wv = align_signals(norm_dv, norm_wv)

print(f"Training lag result: dry={len(processed_dt)} wet={len(processed_wt)}")
print(f"Validation lag result: dry={len(processed_dv)} wet={len(processed_wv)}")

# Plot a short section of aligned signals overlaid
plt.figure(figsize=(12, 4))
plt.plot(processed_dt[int(48000*34.65):int(48000*35)], label='dry aligned', alpha=0.8)
plt.plot(processed_wt[int(48000*34.65):int(48000*35)], label='wet aligned', alpha=0.8)
plt.legend()
plt.title('Aligned signals — training data')
plt.savefig('results/alignment_check.png', dpi=150)
plt.show()

os.makedirs("data/processed", exist_ok=True)
sf.write("data/processed/training_data_10_clean.wav", processed_dt, samplerate=dt_sr)
sf.write("data/processed/training_data_10_driven.wav", processed_wt, samplerate=wt_sr)
sf.write("data/processed/validation_data_10_clean.wav", processed_dv, samplerate=dv_sr)
sf.write("data/processed/validation_data_10_driven.wav", processed_wv, samplerate=wv_sr)







