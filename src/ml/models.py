import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x arrives as (batch, seq_len) — reshape to (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        # Take only the last timestep's output
        return self.fc(out[:, -1]).squeeze(-1)