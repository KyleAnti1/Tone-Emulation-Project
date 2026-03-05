import matplotlib
import numpy as np
import soundfile as sf
import scipy

def load_audio(path):
    """
    :param path: file path to audio file (string)
    :return np.array of signal + sample rate
    """

    audio, sr = sf.read(path) #
    audio = audio.astype(np.float32) # casting
    audio = np.mean(audio, axis=1) # mono

    return audio, sr





