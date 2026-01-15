import torch
import numpy as np
from typing import Dict, List, Optional
from data.messages import PerceptionOutput, TrackedObject, format_objects_to_message

try:
    import accelsight_cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

class MotionTracker:
    """
    Object-level tracking state manager.
    Encapsulates greedy association using spatial distance and Re-ID embeddings.
    """
    def __init__(self, dist_threshold: float = 2.0):
        self.dist_threshold = dist_threshold
        self.next_id = 1
        self.active_tracks: List[TrackedObject] = []

    def update(self, detections: List[TrackedObject]) -> List[TrackedObject]:
        if not self.active_tracks:
            self.active_tracks = [self._create_track(d, is_new=True) for d in detections]
            return self.active_tracks

        updated, matched_det = [], set()
        for track in self.active_tracks:
            best_idx, best_score = -1, -1.0
            for j, det in enumerate(detections):
                if j in matched_det: continue
                dist = np.linalg.norm(track.pos - det.pos)
                if dist < self.dist_threshold:
                    # Score = Cosine Similarity + Proximity
                    sim = (np.dot(track.embedding, det.embedding) / 
                          (np.linalg.norm(track.embedding) * np.linalg.norm(det.embedding) + 1e-6)) if det.embedding is not None else 0
                    score = sim + (1.0 - dist / self.dist_threshold)
                    if score > best_score:
                        best_score, best_idx = score, j

            if best_idx != -1:
                matched_det.add(best_idx)
                updated.append(self._create_track(detections[best_idx], track_id=track.id))

        # Add new tracks
        updated.extend([self._create_track(d, is_new=True) for j, d in enumerate(detections) if j not in matched_det])
        self.active_tracks = updated
        return self.active_tracks

    def _create_track(self, det: TrackedObject, track_id: Optional[int] = None, is_new: bool = False) -> TrackedObject:
        if is_new:
            id_to_use = self.next_id
            self.next_id += 1
        else:
            id_to_use = track_id
        return TrackedObject(id=id_to_use, category=det.category, pos=det.pos, vel=det.vel, 
                             confidence=det.confidence, bbox=det.bbox, embedding=det.embedding)

class PostProcessor:
    """
    AccelSight Perception Engine.
    Orchestrates CUDA-accelerated NMS, 3D coordinate mapping, and object tracking.
    """
    def __init__(self, conf_threshold: float = 0.5, focal_length: float = 500.0, img_res: int = 256, grid_res: int = 16):
        self.conf_threshold = conf_threshold
        self.tracker = MotionTracker()
        self.focal = focal_length
        self.center = img_res / 2.0
        self.scale = img_res / grid_res

    def process(self, outputs: Dict[str, torch.Tensor], timestamp: float) -> List[PerceptionOutput]:
        batch_size = outputs["objectness"].size(0)
        probs = torch.sigmoid(outputs["objectness"])
        
        # Dispatch to CUDA if available, fallback to vectorized CPU logic
        if HAS_CUDA and probs.is_cuda:
            mask = accelsight_cuda.fast_nms(probs, self.conf_threshold)
            world_coords = accelsight_cuda.coordinate_transform(
                outputs["bbox"], torch.ones_like(probs) * 5.0, 
                self.focal, self.focal, self.center, self.center, self.scale, self.scale
            )
        else:
            mask = (probs > self.conf_threshold).float()
            world_coords = torch.zeros((batch_size, 3, *probs.shape[2:]), device=probs.device)

        batch_results = []
        for b in range(batch_size):
            idx = torch.nonzero(mask[b, 0])
            if len(idx) == 0:
                batch_results.append(PerceptionOutput(timestamp, b, [], outputs["controls"][b].cpu().numpy()))
                continue

            # Efficiently extract detected features
            h, w = idx[:, 0], idx[:, 1]
            det_objs = format_objects_to_message(
                ids=np.zeros(len(idx)),
                pos=world_coords[b, :, h, w].T.cpu().numpy(),
                vel=outputs["velocity"][b, :, h, w].T.cpu().numpy(),
                scores=probs[b, 0, h, w].cpu().numpy(),
                bboxes=outputs["bbox"][b, :, h, w].T.cpu().numpy(),
                embeddings=outputs["embedding"][b, :, h, w].T.cpu().numpy()
            )
            
            batch_results.append(PerceptionOutput(timestamp, b, self.tracker.update(det_objs), 
                                                outputs["controls"][b].cpu().numpy()))
        return batch_results
