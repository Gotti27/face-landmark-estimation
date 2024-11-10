import onnx
import torch
from model import FaceLandmark
import onnxoptimizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
model = FaceLandmark()
model = model.to(device)
model.load_state_dict(torch.load('models/weights-1', map_location=device, weights_only=True))

model.eval()
example_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    example_input,
    "models/face-landmark-estimation.onnx",
    export_params=True,
    input_names=["image"],
    output_names=["landmarks"]
)

model_path = "models/face-landmark-estimation.onnx"
model = onnx.load(model_path)
optimized_model = onnxoptimizer.optimize(model)
onnx.save(optimized_model, "models/face-landmark-estimation-optimized.onnx")
