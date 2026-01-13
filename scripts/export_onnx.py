import torch
import torch.onnx
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))
from models.accelsight_net import AccelSightNet

def export_to_onnx(output_path="docs/accelsight_model.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize model
    model = AccelSightNet(num_frames=5)
    model.eval()
    
    # Dummy input: (Batch=1, Frames=5, Channels=3, H=256, W=256)
    dummy_input = torch.randn(1, 5, 3, 256, 256)
    
    # Export
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_frames'],
        output_names=['objectness', 'bbox', 'velocity', 'embedding'],
        dynamic_axes={'input_frames': {0: 'batch_size'}}
    )
    
    if os.path.exists(output_path):
        print("Export successful.")
        # Check ONNX model if onnx is installed
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification successful.")
        except ImportError:
            print("onnx-python not found, skipping checker.")
    else:
        print("Export failed.")

if __name__ == "__main__":
    export_to_onnx()
