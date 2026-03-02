from src.io.audio import load_audio
import numpy as np


clean , sr = load_audio("data/raw/new_set_1142.wav")
print(clean[0:40])
array = np.linspace(0, 44099, 44100)







