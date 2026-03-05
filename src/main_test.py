from src.io.audio import load_audio
from src.io.preprocessing import remove_dc
from src.io.alignment import align_signals
from src.ml.dataset import AudioPairDataset

dry, d_sr = load_audio("data/raw/clean_01_test.wav")
wet, w_sr = load_audio("data/processed/driven_01_test.wav")

dry = remove_dc(dry)
wet = remove_dc(wet)

dry, wet = align_signals(dry, wet)

dataset = AudioPairDataset(dry, wet, window_size=101)
print(len(dataset))
x, y = dataset[0]
print(x.shape, y.shape)