import matplotlib
import numpy as np
import soundfile as sf
import scipy

def load_audio(path):

    audio, sr = sf.read(path) #
    audio = audio.astype(np.float32) # casting
    audio = np.mean(audio, axis=1) # mono

    return audio, sr





