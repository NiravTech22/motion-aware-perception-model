import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.accelsight_net import AccelSightNet
from training.losses import AccelSightMultiTaskLoss

def visualize_inference(data_path="data/sim_data/mock_sample.pt", output_dir="visuals"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run mock_data_gen.py first.")
        return
    
    data = torch.load(data_path)
    frames = data["frames"]
    targets = {
        "gt_objectness": data["gt_objectness"],
        "gt_bbox": data["gt_bbox"],
        "gt_velocity": data["gt_velocity"],
        "gt_ids": data["gt_ids"],
        "gt_controls": data["gt_controls"]
    }
    
    # 2. Run Inference
    model = AccelSightNet(num_frames=5)
    model.eval()
    with torch.no_grad():
        preds = model(frames)
    
    # 3. Compute Losses for plotting
    criterion = AccelSightMultiTaskLoss()
    loss_metrics = criterion(preds, targets)
    
    # --- Plot 1: Loss Breakdown ---
    plt.figure(figsize=(10, 6))
    losses = {k: v.item() for k, v in loss_metrics.items() if k != "total_loss"}
    plt.bar(losses.keys(), losses.values(), color=['skyblue', 'salmon', 'lightgreen', 'orange', 'plum'])
    plt.title(f"Multi-Task Loss Breakdown (Total: {loss_metrics['total_loss'].item():.4f})")
    plt.ylabel("Loss Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "loss_breakdown.png"))
    plt.close()

    # --- Plot 2: Perceptual Heatmaps ---
    # Take the first sample in batch
    idx = 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input Frame (last one in stack)
    img = frames[idx, -1].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min()) # Normalize for display
    axes[0].imshow(img)
    axes[0].set_title("Input Frame (t)")
    axes[0].axis('off')
    
    # GT Objectness
    axes[1].imshow(targets["gt_objectness"][idx, 0].numpy(), cmap='inferno')
    axes[1].set_title("GT Objectness Heatmap")
    axes[1].axis('off')
    
    # Predicted Objectness
    # Apply sigmoid since it's Raw Logits
    pred_obj = torch.sigmoid(preds["objectness"][idx, 0]).numpy()
    axes[2].imshow(pred_obj, cmap='inferno')
    axes[2].set_title("Pred Objectness Heatmap")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perception_heatmaps.png"))
    plt.close()

    # --- Plot 3: Control Signals Comparison ---
    plt.figure(figsize=(8, 5))
    gt_ctrl = targets["gt_controls"][idx].numpy()
    pd_ctrl = preds["controls"][idx].numpy()
    
    x = np.arange(len(gt_ctrl))
    width = 0.35
    
    plt.bar(x - width/2, gt_ctrl, width, label='Ground Truth', color='gray', alpha=0.6)
    plt.bar(x + width/2, pd_ctrl, width, label='Predicted', color='blue', alpha=0.6)
    
    plt.xticks(x, ['Steering', 'Throttle', 'Brake'])
    plt.title("Control Signals Comparison")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "control_signals.png"))
    plt.close()

    print(f"Visualization plots saved to {output_dir}/")

if __name__ == "__main__":
    visualize_inference()
