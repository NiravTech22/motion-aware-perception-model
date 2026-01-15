from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass(frozen=True)
class TrackedObject:
    """
    High-fidelity representation of a tracked instance in 3D world space.
    """
    id: int
    category: int
    pos: np.ndarray        # (3,) [X, Y, Z]
    vel: np.ndarray        # (3,) [VX, VY, VZ]
    confidence: float
    bbox: np.ndarray       # (4,) [dx, dy, dw, dh]
    embedding: Optional[np.ndarray] = None

@dataclass(frozen=True)
class PerceptionOutput:
    """
    Immutable snapshot of the environment state at a specific timestamp.
    """
    timestamp: float
    frame_id: int
    objects: List[TrackedObject] = field(default_factory=list)
    controls: Optional[np.ndarray] = None # [Throttle, Brake, Steering]
    
    def __post_init__(self):
        # Deterministic ordering by ID for downstream consumers (e.g., planners)
        object.__setattr__(self, 'objects', sorted(self.objects, key=lambda x: x.id))

def format_objects_to_message(
    ids: np.ndarray, 
    pos: np.ndarray, 
    vel: np.ndarray, 
    scores: np.ndarray, 
    bboxes: np.ndarray,
    embeddings: Optional[np.ndarray] = None
) -> List[TrackedObject]:
    """
    Factory function for batch-transforming raw detection arrays into semantic objects.
    """
    return [
        TrackedObject(
            id=int(ids[i]),
            category=0, 
            pos=pos[i].copy(),
            vel=vel[i].copy(),
            confidence=float(scores[i]),
            bbox=bboxes[i].copy(),
            embedding=embeddings[i].copy() if embeddings is not None else None
        ) for i in range(len(ids))
    ]
