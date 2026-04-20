import torch
from src.ml.models import MLPModel, LSTMModel
from src.io.audio import load_audio
from src.evaluation.spectral import plot_spectral_comparison

# Load audio
dry, sr = load_audio("data/processed/clean_01_test.wav")
wet, _  = load_audio("data/processed/driven_01_test.wav")

# Load MLP
mlp = MLPModel(input_size=101)
mlp.load_state_dict(torch.load("models/mlp_baseline.pt"))

# Load LSTM
lstm = LSTMModel(input_size=1, hidden_size=32, num_layers=1)
lstm.load_state_dict(torch.load("models/lstm_baseline.pt"))

# Generate comparison
plot_spectral_comparison(
    dry, wet, sr,
    models_dict={"MLP": mlp, "LSTM": lstm}
)