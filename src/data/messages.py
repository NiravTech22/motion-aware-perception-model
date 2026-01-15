from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass(frozen=True)
class TrackedObject:
    """
    Representation of a single tracked object in world coordinates.
    """
    id: int
    category: int
    pos: np.ndarray  # (3,) [X, Y, Z]
    vel: np.ndarray  # (3,) [VX, VY, VZ]
    confidence: float
    bbox: np.ndarray # (4,) [dx, dy, dw, dh] offsets
    embedding: Optional[np.ndarray] = None # Re-ID features

@dataclass(frozen=True)
class PerceptionOutput:
    """
    Consolidated perception message format.
    """
    timestamp: float
    frame_id: int
    objects: List[TrackedObject] = field(default_factory=list)
    controls: Optional[np.ndarray] = None # [Throttle, Brake, Steering]
    
    def __post_init__(self):
        # Ensure determinism: Sort objects by ID
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
    Helper to convert raw arrays into TrackedObject instances.
    Assumes deterministic indexing across all input arrays.
    """
    objects = []
    num_objs = len(ids)
    for i in range(num_objs):
        obj = TrackedObject(
            id=int(ids[i]),
            category=0, # Placeholder for classification
            pos=pos[i].copy(),
            vel=vel[i].copy(),
            confidence=float(scores[i]),
            bbox=bboxes[i].copy(),
            embedding=embeddings[i].copy() if embeddings is not None else None
        )
        objects.append(obj)
    return objects
