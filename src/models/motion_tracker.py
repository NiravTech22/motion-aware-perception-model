import numpy as np
from typing import List, Dict, Optional
from data.messages import TrackedObject

class MotionTracker:
    """
    Object-level tracking state manager.
    Handles frame-to-frame association using spatial distance and Re-ID embeddings.
    Operates strictly on post-NMS detections.
    """
    def __init__(self, dist_threshold: float = 2.0, embed_threshold: float = 0.5):
        self.dist_threshold = dist_threshold
        self.embed_threshold = embed_threshold
        self.next_id = 1
        self.active_tracks: List[TrackedObject] = []

    def _compute_distance(self, obj1: TrackedObject, obj2: TrackedObject) -> float:
        """Euclidean distance between 3D positions."""
        return np.linalg.norm(obj1.pos - obj2.pos)

    def _compute_cosine_sim(self, obj1: TrackedObject, obj2: TrackedObject) -> float:
        """Cosine similarity between Re-ID embeddings."""
        if obj1.embedding is None or obj2.embedding is None:
            return 0.0
        return np.dot(obj1.embedding, obj2.embedding) / (
            np.linalg.norm(obj1.embedding) * np.linalg.norm(obj2.embedding) + 1e-6
        )

    def update(self, detections: List[TrackedObject]) -> List[TrackedObject]:
        """
        Update tracking state with new detections.
        Performs greedy association based on combined spatial and embedding score.
        """
        if not self.active_tracks:
            # Initialize with all detections
            new_tracks = []
            for det in detections:
                new_tracks.append(self._create_new_track(det))
            self.active_tracks = new_tracks
            return self.active_tracks

        # Association matrix
        num_det = len(detections)
        num_track = len(self.active_tracks)
        
        updated_tracks = []
        matched_det_indices = set()
        matched_track_indices = set()

        # Simple greedy association (for incremental implementation)
        # TODO: Implement Hungarian algorithm or more robust association for production
        for i, track in enumerate(self.active_tracks):
            best_match_idx = -1
            best_score = -1.0
            
            for j, det in enumerate(detections):
                if j in matched_det_indices:
                    continue
                
                dist = self._compute_distance(track, det)
                sim = self._compute_cosine_sim(track, det)
                
                # Combined metric: favor spatial closeness but use embedding for verification
                if dist < self.dist_threshold:
                    score = sim + (1.0 - dist / self.dist_threshold)
                    if score > best_score:
                        best_score = score
                        best_match_idx = j

            if best_match_idx != -1:
                # Update track with new detection values but keep ID
                matched_det_indices.add(best_match_idx)
                matched_track_indices.add(i)
                updated_tracks.append(self._update_track(track, detections[best_match_idx]))

        # Handle unmatched detections (new tracks)
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                updated_tracks.append(self._create_new_track(det))

        # Handle unmatched tracks (dying tracks)
        # TODO: Implement track age/cooldown to handle occlusions
        # For now, we only keep matched and new tracks to be deterministic

        self.active_tracks = updated_tracks
        return self.active_tracks

    def _create_new_track(self, det: TrackedObject) -> TrackedObject:
        """Assign a new ID to a detection."""
        obj = TrackedObject(
            id=self.next_id,
            category=det.category,
            pos=det.pos,
            vel=det.vel,
            confidence=det.confidence,
            bbox=det.bbox,
            embedding=det.embedding
        )
        self.next_id += 1
        return obj

    def _update_track(self, old_track: TrackedObject, new_det: TrackedObject) -> TrackedObject:
        """Keep the ID while updating other fields."""
        return TrackedObject(
            id=old_track.id,
            category=new_det.category,
            pos=new_det.pos,
            vel=new_det.vel,
            confidence=new_det.confidence,
            bbox=new_det.bbox,
            embedding=new_det.embedding
        )
