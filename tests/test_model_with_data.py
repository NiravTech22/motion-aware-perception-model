import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.accelsight_net import AccelSightNet
from training.losses import MultiTaskLoss

def test_inference_with_data():
    # 1. Load data
    data_path = "data/sim_data/mock_sample.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    data = torch.load(data_path)
    frames = data["frames"] # (B, 5, 3, 256, 256)
    
    # 2. Initialize model and loss
    model = AccelSightNet(num_frames=5)
    criterion = MultiTaskLoss()
    
    # 3. Forward pass
    print("Running inference...")
    outputs = model(frames)
    
    # 4. Compute Loss
    print("Computing loss...")
    targets = {
        "gt_objectness": data["gt_objectness"],
        "gt_bbox": data["gt_bbox"],
        "gt_velocity": data["gt_velocity"],
        "gt_ids": data["gt_ids"]
    }
    
    loss_metrics = criterion(outputs, targets)
    
    print("\nTest Results:")
    print(f"Total Loss: {loss_metrics['total_loss'].item():.4f}")
    print(f"Objectness Loss: {loss_metrics['obj_loss'].item():.4f}")
    print(f"BBox Loss: {loss_metrics['bbox_loss'].item():.4f}")
    print(f"Velocity Loss: {loss_metrics['vel_loss'].item():.4f}")
    
    print("\nOutput Tensor Shapes:")
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")

if __name__ == "__main__":
    test_inference_with_data()
