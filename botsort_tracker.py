import numpy as np
import cv2
from collections import OrderedDict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

class KalmanFilter:
    """Simple Kalman Filter for tracking"""
    
    def __init__(self):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0, 0, 0],
                                             [0, 0, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 0, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 0, 0, 1, 0, 0],
                                           [0, 1, 0, 0, 0, 1, 0],
                                           [0, 0, 1, 0, 0, 0, 1],
                                           [0, 0, 0, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        
    def init(self, bbox):
        """Initialize Kalman filter with bounding box"""
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        self.kf.statePre = np.array([cx, cy, w, h, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0], dtype=np.float32)
        
    def predict(self):
        """Predict next state"""
        self.kf.predict()
        state = self.kf.statePre
        cx, cy, w, h = state[0], state[1], state[2], state[3]
        return np.array([cx - w/2, cy - h/2, w, h])
        
    def update(self, bbox):
        """Update with measurement"""
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        measurement = np.array([cx, cy, w, h], dtype=np.float32)
        self.kf.correct(measurement)

class Track:
    """Track class for Bot-SORT"""
    
    def __init__(self, track_id, bbox, score, frame_id):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.score = score
        self.frame_id = frame_id
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        
        # Initialize Kalman filter
        self.kf = KalmanFilter()
        self.kf.init(bbox)
        
        # Appearance features (placeholder)
        self.features = None
        
    def predict(self):
        """Predict next state"""
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.predict()
        
    def update(self, bbox, score):
        """Update track with new detection"""
        self.bbox = bbox
        self.score = score
        self.kf.update(bbox)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
    def is_confirmed(self):
        """Check if track is confirmed"""
        return self.hits >= 3
        
    def is_deleted(self):
        """Check if track should be deleted"""
        return self.time_since_update > 30

class BotSORT:
    """Bot-SORT tracker implementation"""
    
    def __init__(self, track_thresh=0.5, match_thresh=0.8, frame_rate=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.tracks = []
        self.next_id = 1
        self.frame_id = 0
        
    def update(self, detections, frame):
        """Update tracker with new detections"""
        self.frame_id += 1
        
        # Convert detections to [x, y, w, h, score] format
        dets = []
        for det in detections:
            if isinstance(det, dict):
                x1, y1, x2, y2 = det['bbox']
                score = det['confidence']
                w, h = x2 - x1, y2 - y1
                dets.append([x1, y1, w, h, score])
            else:
                # det is already in [x, y, w, h, score] format
                dets.append(det)
        
        dets = np.array(dets)
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
            
        # Separate confirmed and unconfirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        unconfirmed_tracks = [t for t in self.tracks if not t.is_confirmed()]
        
        # Associate detections with confirmed tracks
        if len(confirmed_tracks) > 0 and len(dets) > 0:
            matches, unmatched_dets, unmatched_tracks = self.associate_detections_to_trackers(
                dets, confirmed_tracks
            )
            
            # Update matched tracks
            for det_idx, track_idx in matches:
                track = confirmed_tracks[track_idx]
                det = dets[det_idx]
                track.update(det[:4], det[4])
                
            # Handle unmatched detections and tracks
            unmatched_detections = dets[unmatched_dets]
            unmatched_trackers = [confirmed_tracks[i] for i in unmatched_tracks]
        else:
            unmatched_detections = dets
            unmatched_trackers = confirmed_tracks
            
        # Associate unmatched detections with unconfirmed tracks
        if len(unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
            matches, unmatched_dets, unmatched_tracks = self.associate_detections_to_trackers(
                unmatched_detections, unconfirmed_tracks
            )
            
            # Update matched unconfirmed tracks
            for det_idx, track_idx in matches:
                track = unconfirmed_tracks[track_idx]
                det = unmatched_detections[det_idx]
                track.update(det[:4], det[4])
                
            # Remaining unmatched detections
            unmatched_detections = unmatched_detections[unmatched_dets]
            
        # Create new tracks for remaining unmatched detections
        for det in unmatched_detections:
            if det[4] > self.track_thresh:
                track = Track(self.next_id, det[:4], det[4], self.frame_id)
                self.tracks.append(track)
                self.next_id += 1
                
        # Remove tracks that are too old
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return tracked objects
        tracked_objects = []
        for track in self.tracks:
            if track.is_confirmed():
                bbox = track.bbox
                tracked_objects.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], track.track_id])
                
        return tracked_objects
        
    def associate_detections_to_trackers(self, detections, trackers):
        """Associate detections with trackers using IoU"""
        if len(trackers) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(trackers)))
            
        # Calculate IoU matrix
        iou_matrix = self.calculate_iou_matrix(detections, trackers)
        
        # Use Hungarian algorithm for assignment
        det_indices, track_indices = linear_sum_assignment(-iou_matrix)
        
        matches = []
        unmatched_dets = []
        unmatched_tracks = []
        
        # Filter matches based on IoU threshold
        for det_idx, track_idx in zip(det_indices, track_indices):
            if iou_matrix[det_idx, track_idx] > self.match_thresh:
                matches.append((det_idx, track_idx))
            else:
                unmatched_dets.append(det_idx)
                unmatched_tracks.append(track_idx)
                
        # Find unmatched detections and tracks
        for i in range(len(detections)):
            if i not in [m[0] for m in matches]:
                unmatched_dets.append(i)
                
        for i in range(len(trackers)):
            if i not in [m[1] for m in matches]:
                unmatched_tracks.append(i)
                
        return matches, unmatched_dets, unmatched_tracks
        
    def calculate_iou_matrix(self, detections, trackers):
        """Calculate IoU matrix between detections and trackers"""
        iou_matrix = np.zeros((len(detections), len(trackers)))
        
        for i, det in enumerate(detections):
            for j, tracker in enumerate(trackers):
                iou_matrix[i, j] = self.calculate_iou(det[:4], tracker.bbox)
                
        return iou_matrix
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
