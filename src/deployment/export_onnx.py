import torch
from src.ml.models import LSTMModel

# Load trained model
model = LSTMModel(input_size=1, hidden_size=32, num_layers=1)
model.load_state_dict(torch.load("models/lstm_baseline.pt"))
model.eval()

# Dummy input matching your window size
dummy_input = torch.randn(1, 101)

torch.onnx.export(
    model,
    dummy_input,
    "src/deployment/onnx/lstm_overdrive.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=14,  # stable version, no conversion issues
    dynamo=False       # use legacy exporter, avoids onnxscript conflicts
)

print("Export successful")