import torch
import os

def generate_mock_data(output_dir="data/sim_data", num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    
    frames = torch.randn(num_samples, 5, 3, 256, 256)
    
    # Ground Truth at the bottleneck resolution (1/16 of 256 = 16)
    gt_objectness = (torch.rand(num_samples, 1, 16, 16) > 0.8).float()
    gt_bbox = torch.randn(num_samples, 4, 16, 16)
    gt_velocity = torch.randn(num_samples, 3, 16, 16)
    gt_ids = torch.randint(0, 10, (num_samples, 16, 16))
    
    data = {
        "frames": frames,
        "gt_objectness": gt_objectness,
        "gt_bbox": gt_bbox,
        "gt_velocity": gt_velocity,
        "gt_ids": gt_ids
    }
    
    torch.save(data, os.path.join(output_dir, "mock_sample.pt"))
    print(f"Generated mock data at {output_dir}/mock_sample.pt")

if __name__ == "__main__":
    generate_mock_data()
