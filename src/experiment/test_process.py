from src.io.alignment import align_signals
from src.io.audio import load_audio
from src.io.preprocessing import *
import soundfile as sf

dry, d_sr = load_audio("data/raw/clean_01_test.wav")
wet, w_sr = load_audio("data/raw/driven_01_test.wav")

dc_dry = remove_dc(dry)
dc_wet = remove_dc(wet)

norm_dry = normalise(dc_dry)
norm_wet = normalise(dc_wet)

processed_dry, processed_wet = align_signals(norm_dry, norm_wet)

#sf.write("data/processed/clean_01_test.wav", processed_dry, samplerate=d_sr)
#sf.write("data/processed/driven_01_test.wav", processed_wet, samplerate=w_sr)







