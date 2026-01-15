import torch
import numpy as np
from typing import Dict, List, Optional
from data.messages import PerceptionOutput, TrackedObject, format_objects_to_message
from .motion_tracker import MotionTracker

try:
    # Attempt to import compiled CUDA ops
    import accelsight_cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

class PostProcessor:
    """
    Orchestrates the decoding of raw neural inference outputs into perception messages.
    Sole consumer of TensorRT tensors.
    """
    def __init__(
        self, 
        conf_threshold: float = 0.5,
        focal_length: float = 500.0,
        img_res: int = 256,
        grid_res: int = 16
    ):
        self.conf_threshold = conf_threshold
        self.tracker = MotionTracker()
        
        # Camera intrinsics (Placeholders - should be passed in production)
        self.focal_x = focal_length
        self.focal_y = focal_length
        self.center_x = img_res / 2.0
        self.center_y = img_res / 2.0
        self.grid_scale = img_res / grid_res

    def process(self, outputs: Dict[str, torch.Tensor], timestamp: float) -> List[PerceptionOutput]:
        """
        Process a batch of network outputs.
        outputs keys: objectness, bbox, velocity, embedding, controls
        """
        batch_size = outputs["objectness"].size(0)
        results = []

        # 1. Apply Sigmoid to objectness if raw logits
        # TODO: Confirm if TensorRT export includes sigmoid. Assuming raw for now.
        probs = torch.sigmoid(outputs["objectness"])

        # 2. CUDA-Accelerated NMS (if available)
        if HAS_CUDA and probs.is_cuda:
            mask = accelsight_cuda.fast_nms(probs, self.conf_threshold)
            # 3. Coordinate Transform (if available)
            # Placeholder for depth_map; using Z=5.0 for demo if not provided
            depth = torch.ones_like(probs) * 5.0 
            world_coords = accelsight_cuda.coordinate_transform(
                outputs["bbox"], depth, 
                self.focal_x, self.focal_y, 
                self.center_x, self.center_y,
                self.grid_scale, self.grid_scale
            )
        else:
            # CPU Fallback / Non-CUDA logic
            # Simplified: Find local maxima above threshold
            mask = (probs > self.conf_threshold).float()
            # In a real system, we'd do a 3x3 max pool comparison here on CPU as well
            
            # Placeholder for world_coords on CPU
            world_coords = torch.zeros((batch_size, 3, probs.size(2), probs.size(3)), device=probs.device)

        # 4. Extract and Track (Per-image in batch)
        # Note: We must avoid Python loops over GRID CELLS, but we loop over IMAGES in batch.
        for b in range(batch_size):
            img_mask = mask[b, 0]
            indices = torch.nonzero(img_mask) # (N, 2) [H_idx, W_idx]
            
            if len(indices) == 0:
                results.append(PerceptionOutput(
                    timestamp=timestamp,
                    frame_id=b,
                    objects=[],
                    controls=outputs["controls"][b].detach().cpu().numpy()
                ))
                continue

            # Gather attributes for detections
            h_idx = indices[:, 0]
            w_idx = indices[:, 1]
            
            scores = probs[b, 0, h_idx, w_idx].detach().cpu().numpy()
            pos = world_coords[b, :, h_idx, w_idx].permute(1, 0).detach().cpu().numpy()
            vel = outputs["velocity"][b, :, h_idx, w_idx].permute(1, 0).detach().cpu().numpy()
            bboxes = outputs["bbox"][b, :, h_idx, w_idx].permute(1, 0).detach().cpu().numpy()
            embeds = outputs["embedding"][b, :, h_idx, w_idx].permute(1, 0).detach().cpu().numpy()

            # Create discrete detection objects
            ids_placeholder = np.zeros(len(scores)) # Tracker will assign real IDs
            detections = format_objects_to_message(
                ids=ids_placeholder,
                pos=pos,
                vel=vel,
                scores=scores,
                bboxes=bboxes,
                embeddings=embeds
            )

            # 5. Tracking association
            tracked_objects = self.tracker.update(detections)

            results.append(PerceptionOutput(
                timestamp=timestamp,
                frame_id=b,
                objects=tracked_objects,
                controls=outputs["controls"][b].detach().cpu().numpy()
            ))

        return results
