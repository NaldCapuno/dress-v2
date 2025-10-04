import numpy as np
import cv2
from collections import OrderedDict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

# =============================================================================
# KALMAN FILTER CLASS
# =============================================================================

class KalmanFilter:
    """Enhanced Kalman Filter for bounding box tracking with velocity estimation"""
    
    def __init__(self, frame_rate=30):
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        # Measurement: [cx, cy, w, h]
        self.kf = cv2.KalmanFilter(8, 4)
        self.frame_rate = max(frame_rate, 1)  # Ensure frame rate is at least 1
        self.dt = 1.0 / self.frame_rate  # Time step
        
        # Measurement matrix (we observe position and size)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Transition matrix (constant velocity model with time step)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, self.dt, 0, 0, 0],      # cx = cx + vx * dt
            [0, 1, 0, 0, 0, self.dt, 0, 0],      # cy = cy + vy * dt
            [0, 0, 1, 0, 0, 0, self.dt, 0],      # w = w + vw * dt
            [0, 0, 0, 1, 0, 0, 0, self.dt],      # h = h + vh * dt
            [0, 0, 0, 0, 1, 0, 0, 0],            # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],            # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],            # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]             # vh = vh
        ], dtype=np.float32)
        
        # Process noise covariance (how much we trust our model)
        self.noise = 0.02  # Reduced noise for better tracking
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * self.noise
        
        # Measurement noise covariance (how much we trust measurements)
        measurement_noise = 0.1  # Lower noise for more responsive tracking
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        
        # Error covariance (initial uncertainty)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1000
        
        self.initialized = False
        
    def init(self, bbox):
        """Initialize Kalman filter with bounding box [x, y, w, h ]"""
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        
        # Initial state: [cx, cy, w, h, 0, 0, 0, 0] with higher velocity uncertainty
        initial_state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        self.kf.statePre = initial_state
        self.kf.statePost = initial_state
        
        # Set higher initial velocity uncertainty for faster adaptation
        self.kf.errorCovPost[4:, 4:] = np.eye(4, dtype=np.float32) * 100
        
        self.initialized = True
        
    def predict(self):
        """Predict next state and return predicted bounding box"""
        if not self.initialized:
            return None
            
        self.kf.predict()
        state = self.kf.statePre
        cx, cy, w, h = state[0], state[1], state[2], state[3]
        
        # Ensure reasonable bounds for width and height
        w = max(1.0, w)
        h = max(1.0, h)
        
        # Convert center coordinates back to top-left coordinates
        x, y = cx - w/2, cy - h/2
        
        return np.array([x, y, w, h])
        
    def update(self, bbox):
        """Update with measurement [x, y, w, h]"""
        if not self.initialized:
            self.init(bbox)
            return
            
        x, y, w, h = bbox
        
        # Ensure valid bounding box
        w = max(1.0, w)
        h = max(1.0, h)
        
        cx, cy = x + w/2, y + h/2
        measurement = np.array([cx, cy, w, h], dtype=np.float32)
        
        self.kf.correct(measurement)
        
    def get_state(self):
        """Get current state mean and covariance"""
        if not self.initialized:
            return None, None
            
        state_mean = self.kf.statePost[:4]  # [cx, cy, w, h]
        state_cov = self.kf.errorCovPost[:4, :4]  # 4x4 covariance matrix
        
        return state_mean, state_cov

# =============================================================================
# MAHALANOBIS DISTANCE CALCULATOR CLASS
# =============================================================================

class MahalanobisCalculator:
    """Calculator for Mahalanobis distance between detection and track state"""
    
    def __init__(self, regularization=1e-6, max_distance=50.0):
        self.regularization = regularization
        self.max_distance = max_distance
    
    def calculate_distance(self, detection_center, track_state_mean, track_state_cov):
        """Calculate Mahalanobis distance between detection and track"""
        if track_state_mean is None or track_state_cov is None:
            return self.max_distance
        
        # Calculate Mahalanobis distance
        diff = detection_center - track_state_mean
        # Ensure covariance is invertible
        state_cov = track_state_cov + np.eye(4) * self.regularization
        
        try:
            inv_cov = np.linalg.inv(state_cov)
            mahalanobis_dist = np.sqrt(max(0, diff.T @ inv_cov @ diff))
            mahalanobis_dist = min(mahalanobis_dist, self.max_distance)
        except np.linalg.LinAlgError:
            mahalanobis_dist = self.max_distance
        
        return mahalanobis_dist

# =============================================================================
# COSINE SIMILARITY CALCULATOR CLASS
# =============================================================================

class CosineSimilarityCalculator:
    """Calculator for cosine similarity between detection and track features"""
    
    def __init__(self, lambda_weight=0.01):
        self.lambda_weight = lambda_weight
    
    def extract_features(self, bbox):
        """Extract simple appearance features from bounding box [x, y, w, h]"""
        x, y, w, h = bbox
        aspect_ratio = w / h if h > 0 else 1.0
        area_normalized = (w * h) / 100000
        
        features = np.array([aspect_ratio, area_normalized])
        return features
    
    def calculate_similarity(self, detection_bbox, track_features):
        """Calculate cosine similarity between detection and track features"""
        # Extract detection features
        det_features = self.extract_features(detection_bbox)
        
        # Get track features (should be pre-computed average/history)
        if track_features is None or len(track_features) == 0:
            return 0.0
        
        # Calculate cosine similarity
        det_norm = np.linalg.norm(det_features)
        track_norm = np.linalg.norm(track_features)
        
        if det_norm == 0 or track_norm == 0:
            return 0.0
            
        cosine_sim = np.dot(det_features, track_features) / (det_norm * track_norm)
        
        # Return similarity score [0, 1]
        return max(0.0, cosine_sim)
    
    def get_weighted_cost(self, mahalanobis_dist, cosine_sim):
        """Combine Mahalanobis distance with cosine similarity"""
        return mahalanobis_dist - self.lambda_weight * cosine_sim

# =============================================================================
# IOU CALCULATOR CLASS
# =============================================================================

class IoUCalculator:
    """Calculator for Intersection over Union (IoU) between bounding boxes"""
    
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes [x, y, w, h]"""
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
    
    def calculate_iou_matrix(self, detections, tracks):
        """Calculate IoU between all detection-track pairs"""
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                iou_matrix[i, j] = self.calculate_iou(det[:4], track.bbox)
        return iou_matrix
    
    def associate_with_iou(self, detections, tracks):
        """Associate detections with tracks using IoU and Hungarian algorithm"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Calculate IoU matrix
        iou_matrix = self.calculate_iou_matrix(detections, tracks)
        
        # Use Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(1 - iou_matrix)  # Minimize (1-IoU)
        
        matches = []
        matched_dets = set()
        matched_tracks = set()
        
        # Filter matches based on IoU threshold
        for row_idx, col_idx in zip(row_indices, col_indices):
            if row_idx < len(detections) and col_idx < len(tracks):
                iou = iou_matrix[row_idx, col_idx]
                if iou > self.iou_threshold:
                    matches.append((row_idx, col_idx))
                    matched_dets.add(row_idx)
                    matched_tracks.add(col_idx)
        
        # Find unmatched
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        
        return matches, unmatched_dets, unmatched_tracks

# =============================================================================
# TRACK CLASS
# =============================================================================

class Track:
    """Enhanced Track class for Bot-SORT with appearance features"""
    
    def __init__(self, track_id, bbox, score, frame_id):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.score = score
        self.frame_id = frame_id
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.confirmed = False
        
        # Initialize Kalman filter with frame rate
        self.kf = KalmanFilter(frame_rate=30)
        self.kf.init(bbox)
        
        # Initialize feature calculator
        self.feature_calculator = CosineSimilarityCalculator()
        self.features = self.feature_calculator.extract_features(bbox)
        
        # Track history for cosine similarity
        self.feature_history = []
        self.feature_history.append(self.features.copy())
        
        # Store last bbox for velocity calculation
        self._last_bbox = bbox.copy()
        
    def predict(self):
        """Predict next state"""
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        predicted_bbox = self.kf.predict()
        if predicted_bbox is not None:
            self.bbox = predicted_bbox
        
        return self.bbox
        
    def update(self, bbox, score):
        """Update track with new detection"""
        # Calculate velocity from position change for faster adaptation
        if self.time_since_update == 0 and hasattr(self, '_last_bbox'):
            # Estimate velocity from position change
            old_cx = self._last_bbox[0] + self._last_bbox[2]/2
            old_cy = self._last_bbox[1] + self._last_bbox[3]/2
            new_cx = bbox[0] + bbox[2]/2
            new_cy = bbox[1] + bbox[3]/2
            
            # Update velocity in Kalman filter state for faster adaptation
            if hasattr(self.kf, 'statePost'):
                self.kf.statePost[4] = (new_cx - old_cx) * 30  # Scale by frame rate
                self.kf.statePost[5] = (new_cy - old_cy) * 30
        
        self._last_bbox = bbox.copy()
        self.bbox = bbox
        self.score = score
        self.kf.update(bbox)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        # Update appearance features
        self.features = self.feature_calculator.extract_features(bbox)
        
        # Add to feature history
        self.feature_history.append(self.features.copy())
        if len(self.feature_history) > 10:  # Keep last 10 features
            self.feature_history.pop(0)
            
        # Confirm track after getting enough updates (immediate for better tracking)
        if self.hits >= 1 and not self.confirmed:
            self.confirmed = True
        
    def get_last_feature(self):
        """Get last appearance feature"""
        return self.features
        
    def get_feature_history(self):
        """Get feature history for cosine similarity"""
        if len(self.feature_history) > 0:
            return np.mean(self.feature_history, axis=0)
        return self.features
        
    def is_confirmed(self):
        """Check if track is confirmed"""
        return self.confirmed
        
    def is_deleted(self):
        """Check if track should be deleted"""
        return self.time_since_update > 30

# =============================================================================
# BOTSORT TRACKER CLASS (refactored to use separated algorithms)
# =============================================================================

class BotSORT:
    """Rewritten Bot-SORT tracker using only Kalman filter, Mahalanobis distance, and cosine similarity"""
    
    def __init__(self, track_thresh=0.5, match_thresh=5.0, frame_rate=30, 
                 mahalanobis_thresh=2.0, cosine_thresh=0.01):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.mahalanobis_thresh = mahalanobis_thresh
        self.cosine_thresh = cosine_thresh
        
        self.tracks = []
        self.next_id = 1
        self.frame_id = 0
        
        # Initialize algorithm calculators
        self.mahalanobis_calc = MahalanobisCalculator()
        self.cosine_calc = CosineSimilarityCalculator(lambda_weight=0.01)
        self.iou_calc = IoUCalculator()
        
    def update(self, detections, frame=None):
        """Update tracker with new detections"""
        self.frame_id += 1
        
        # Convert detections to [x, y, w, h, score] format
        dets = self._convert_detections(detections)
        
        # Predict existing tracks
        for track in self.tracks:
            track.predict()
            
        # Associate detections with tracks using Mahalanobis distance + cosine similarity
        if len(dets) > 0 and len(self.tracks) > 0:
            # Calculate association cost matrix
            cost_matrix = self._calculate_cost_matrix(dets, self.tracks)
            
            # Apply Hungarian algorithm for optimal assignment
            if cost_matrix.size > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                # Filter matches based on thresholds - but be more lenient
                matches, unmatched_dets, unmatched_tracks = self._filter_matches(
                    row_indices, col_indices, cost_matrix, dets, self.tracks
                )
                
                # Always try IoU association as supplementary method
                if len(dets) > 0 and len(self.tracks) > 0:
                    iou_matches, _, _ = self.iou_calc.associate_with_iou(dets, self.tracks)
                    
                    # Add IoU matches that weren't captured by Mahal+cosine
                    existing_matched_dets = set(m[0] for m in matches)
                    for det_idx, track_idx in iou_matches:
                        if det_idx not in existing_matched_dets:
                            matched = False
                            # Check if this track is already matched
                            for existing_det_idx, existing_track_idx in matches:
                                if existing_track_idx == track_idx:
                                    matched = True
                                    break
                            if not matched:
                                matches.append((det_idx, track_idx))
                
                # Update unmatched lists based on final matches
                matched_det_set = set(m[0] for m in matches)
                matched_track_set = set(m[1] for m in matches)
                unmatched_dets = [i for i in range(len(dets)) if i not in matched_det_set]
                unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_track_set]
            else:
                matches = []
                unmatched_dets = list(range(len(dets)))
                unmatched_tracks = list(range(len(self.tracks)))
        else:
            matches = []
            unmatched_dets = list(range(len(dets)))
            unmatched_tracks = list(range(len(self.tracks)))
        
        # Store track IDs for detections BEFORE modifying tracks list
        det_to_track = {}
        for det_idx, track_idx in matches:
            det_to_track[det_idx] = self.tracks[track_idx].track_id
            
        # Update matched tracks
        for det_idx, track_idx in matches:
            track = self.tracks[track_idx]
            det = dets[det_idx]
            track.update(det[:4], det[4])
            
        # Remove tracks that are too old BEFORE adding new ones
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Create new tracks for unmatched detections (immediate ID assignment)
        for det_idx in unmatched_dets:
            if len(dets) > det_idx and dets[det_idx][4] > self.track_thresh:
                track = Track(self.next_id, dets[det_idx][:4], dets[det_idx][4], self.frame_id)
                self.tracks.append(track)
                det_to_track[det_idx] = self.next_id
                self.next_id += 1
        
        # Return tracked objects, one per input detection
        tracked_objects = []
        for i in range(len(dets)):
            if i in det_to_track:
                # Find the track for this detection
                track_id = det_to_track[i]
                for track in self.tracks:
                    if track.track_id == track_id:
                        bbox = track.bbox
                        tracked_objects.append([
                            bbox[0], bbox[1], bbox[2], bbox[3], track.track_id
                        ])
                        break
                else:
                    # Fallback to raw detection
                    det = dets[i]
                    tracked_objects.append([det[0], det[1], det[2], det[3], 0])
            else:
                # No track assigned, return raw detection
                det = dets[i]
                tracked_objects.append([det[0], det[1], det[2], det[3], 0])
                
        return tracked_objects
        
    def _convert_detections(self, detections):
        """Convert detections to standard format [x, y, w, h, score]"""
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
        
        return np.array(dets)
        
    def _calculate_cost_matrix(self, detections, tracks):
        """Calculate cost matrix using Mahalanobis distance + cosine similarity"""
        if len(tracks) == 0 or len(detections) == 0:
            return np.array([])
            
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            det_center = np.array([det[0] + det[2]/2, det[1] + det[3]/2, det[2], det[3]])
            
            for j, track in enumerate(tracks):
                # Get track state and covariance
                state_mean, state_cov = track.kf.get_state()
                
                # Calculate Mahalanobis distance
                mahalanobis_dist = self.mahalanobis_calc.calculate_distance(det_center, state_mean, state_cov)
                
                # Calculate cosine similarity for appearance features
                track_features = track.get_feature_history()
                cos_sim = self.cosine_calc.calculate_similarity(det[:4], track_features)
                
                # Combine costs: Mahalanobis distance - lambda * cosine similarity
                cost = self.cosine_calc.get_weighted_cost(mahalanobis_dist, cos_sim)
                cost_matrix[i, j] = cost
                    
        return cost_matrix
        
    def _filter_matches(self, row_indices, col_indices, cost_matrix, detections, tracks):
        """Filter matches based on Mahalanobis and cosine similarity thresholds"""
        matches = []
        unmatched_dets = []
        unmatched_tracks = []
        
        matched_dets = set()
        matched_tracks = set()
        
        # Process valid matches
        for row_idx, col_idx in zip(row_indices, col_indices):
            if row_idx < len(detections) and col_idx < len(tracks):
                cost = cost_matrix[row_idx, col_idx]
                
                # Check both Mahalanobis distance and cosine similarity thresholds
                track_features = tracks[col_idx].get_feature_history()
                cs_sim = self.cosine_calc.calculate_similarity(detections[row_idx][:4], track_features)
                
                # Apply thresholds
                if (cost < self.match_thresh and 
                    cs_sim > self.cosine_thresh):
                    
                    matches.append((row_idx, col_idx))
                    matched_dets.add(row_idx)
                    matched_tracks.add(col_idx)
        
        # Find unmatched
        for i in range(len(detections)):
            if i not in matched_dets:
                unmatched_dets.append(i)
                
        for i in range(len(tracks)):
            if i not in matched_tracks:
                unmatched_tracks.append(i)
                
        return matches, unmatched_dets, unmatched_tracks
        
    def get_active_track_count(self):
        """Get number of active tracks"""
        return len([t for t in self.tracks if not t.is_deleted()])
        
    def get_confirmed_track_count(self):
        """Get number of confirmed tracks"""
        return len([t for t in self.tracks if t.is_confirmed()])