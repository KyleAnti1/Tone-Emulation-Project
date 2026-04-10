import torch
from src.ml.models import MLPModel

# Load trained model
model = MLPModel(input_size=101)
model.load_state_dict(torch.load("models/mlp_baseline.pt"))
model.eval()

# Dummy input matching your window size
dummy_input = torch.randn(1, 101)

torch.onnx.export(
    model,
    dummy_input,
    "src/deployment/onnx/mlp_overdrive.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=14,  # stable version, no conversion issues
    dynamo=False       # use legacy exporter, avoids onnxscript conflicts
)

print("Export successful")