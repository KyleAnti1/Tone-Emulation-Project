import numpy as np
import soundfile as sf

def load_audio(path):

    audio, sr = sf.read(path)
    audio = audio.astype(np.float32) # casting to float


    if audio.ndim > 1: # convert to mono if stereo
        audio = np.mean(audio, axis=1)

    return audio, sr





