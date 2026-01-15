import torch
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))

from models.post_processor import PostProcessor
from data.messages import PerceptionOutput

def verify_phase4():
    print("--- Phase 4 Verification: Motion-Aware Logic ---")
    
    # 1. Initialize PostProcessor
    # Mocking images of 256x256, heatmap of 16x16
    post_proc = PostProcessor(conf_threshold=0.5, img_res=256, grid_res=16)
    
    # 2. Simulate Frame 1
    print("Simulating Frame 1...")
    # Mock outputs: 1 batch, 16x16 grid
    # Place an object at (5, 5)
    obj_feat1 = torch.zeros(1, 1, 16, 16)
    obj_feat1[0, 0, 5, 5] = 10.0 # High logit for high confidence
    
    bbox_feat1 = torch.zeros(1, 4, 16, 16)
    # dx=2.0, dy=2.0, dw=10.0, dh=10.0
    bbox_feat1[0, :, 5, 5] = torch.tensor([2.0, 2.0, 10.0, 10.0])
    
    vel_feat1 = torch.zeros(1, 3, 16, 16)
    vel_feat1[0, :, 5, 5] = torch.tensor([0.1, 0.0, 0.0])
    
    embed_feat1 = torch.randn(1, 128, 16, 16)
    # Sample embedding for Object A
    embed_A = torch.randn(128)
    embed_A = embed_A / torch.norm(embed_A)
    embed_feat1[0, :, 5, 5] = embed_A
    
    ctrl_feat1 = torch.zeros(1, 3)
    
    outputs1 = {
        "objectness": obj_feat1,
        "bbox": bbox_feat1,
        "velocity": vel_feat1,
        "embedding": embed_feat1,
        "controls": ctrl_feat1
    }
    
    res1 = post_proc.process(outputs1, timestamp=0.0)
    
    assert len(res1[0].objects) == 1, "Should detect 1 object in Frame 1"
    tracker_id = res1[0].objects[0].id
    print(f"Frame 1: Detected Object ID {tracker_id} at grid (5, 5)")

    # 3. Simulate Frame 2 (Object moves slightly, same embedding)
    print("Simulating Frame 2...")
    obj_feat2 = torch.zeros(1, 1, 16, 16)
    obj_feat2[0, 0, 5, 6] = 10.0 # Moved from (5,5) to (5,6)
    
    bbox_feat2 = torch.zeros(1, 4, 16, 16)
    bbox_feat2[0, :, 5, 6] = torch.tensor([2.0, 3.0, 10.0, 10.0]) # Slight bbox shift
    
    vel_feat2 = torch.zeros(1, 3, 16, 16)
    vel_feat2[0, :, 5, 6] = torch.tensor([0.15, 0.0, 0.0])
    
    embed_feat2 = torch.randn(1, 128, 16, 16)
    embed_feat2[0, :, 5, 6] = embed_A # EXACT same embedding
    
    outputs2 = {
        "objectness": obj_feat2,
        "bbox": bbox_feat2,
        "velocity": vel_feat2,
        "embedding": embed_feat2,
        "controls": ctrl_feat1
    }
    
    res2 = post_proc.process(outputs2, timestamp=0.1)
    
    assert len(res2[0].objects) == 1, "Should detect 1 object in Frame 2"
    tracker_id2 = res2[0].objects[0].id
    
    print(f"Frame 2: Detected Object ID {tracker_id2} at grid (5, 6)")
    
    # 4. Assert Persistence
    assert tracker_id == tracker_id2, f"ID mismatch: {tracker_id} vs {tracker_id2}. Tracking failed."
    
    # 5. Verify Output Format
    assert isinstance(res2[0], PerceptionOutput), "Result should be PerceptionOutput"
    assert res2[0].timestamp == 0.1, "Timestamp mismatch"
    
    print("\nPhase 4 Motion-Aware Verification: PASSED")

if __name__ == "__main__":
    verify_phase4()
