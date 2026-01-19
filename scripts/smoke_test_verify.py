import sys, os
sys.path.append(os.path.abspath("src"))
import torch
from models.accelsight_net import AccelSightNet

try:
    print("Attempting with defaults...")
    model = AccelSightNet(num_frames=5) # Default input_channels=3
    x = torch.randn(1, 5, 3, 224, 224) # B, T, C, H, W
    y = model(x)
    print("Success with defaults (3 channels).")
except Exception as e:
    print(f"Failed with defaults: {e}")

try:
    print("Attempting with 5 channels...")
    model5 = AccelSightNet(num_frames=5, input_channels=5)
    x5 = torch.randn(1, 5, 5, 224, 224) 
    y5 = model5(x5)
    print("Success with 5 channels.")
except Exception as e:
    print(f"Failed with 5 channels: {e}")
