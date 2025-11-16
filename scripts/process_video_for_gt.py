"""
Process video with BOTSORT tracker and export tracking results in MOT format.
This generates initial annotations that can be manually corrected.
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.botsort_tracker import BotSORT


def detect_persons(frame, person_model):
    """Detect persons in frame using YOLO"""
    results = person_model(frame)
    detections = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Check if it's a person (class 0 in COCO dataset)
                if class_id == 0 and confidence > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class': 'person'
                    })
    
    return detections


def process_video(video_path, output_dir=None):
    """
    Process video with BOTSORT and export tracking results in MOT format.
    
    MOT format: <frame_id>,<object_id>,<x>,<y>,<width>,<height>,<confidence>,<x_3d>,<y_3d>,<z_3d>
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), 'gt_annotations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading YOLO person detection model...")
    person_model = YOLO('models/yolov8n.pt')
    
    # Initialize tracker
    tracker = BotSORT()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Output files
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    mot_file = os.path.join(output_dir, f"{video_name}_botsort.txt")
    visualization_dir = os.path.join(output_dir, f"{video_name}_visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Process video frame by frame
    frame_id = 0
    tracking_results = []
    
    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # Detect persons
        detections = detect_persons(frame, person_model)
        
        # Convert to tracker format [x, y, w, h, confidence]
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            dets.append([x1, y1, w, h, det['confidence']])
        
        if len(dets) > 0:
            dets = np.array(dets)
            # Update tracker
            tracked_objects = tracker.update(dets, frame)
            
            # Save tracking results in MOT format
            for track in tracked_objects:
                x, y, w, h, track_id = track[0], track[1], track[2], track[3], int(track[4])
                # MOT format: frame_id, object_id, x, y, width, height, confidence, x_3d, y_3d, z_3d
                # confidence is set to 1.0 for ground truth (you'll correct this manually)
                tracking_results.append({
                    'frame_id': frame_id,
                    'object_id': track_id,
                    'bbox': [x, y, w, h],
                    'confidence': 1.0
                })
            
            # Create visualization frame
            vis_frame = frame.copy()
            for track in tracked_objects:
                x, y, w, h, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
                x2, y2 = x + w, y + h
                
                # Draw bounding box
                color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)
                cv2.rectangle(vis_frame, (x, y), (x2, y2), color, 2)
                
                # Draw track ID
                label = f"ID:{track_id}"
                cv2.putText(vis_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Save visualization frame (every 10th frame to save space)
            if frame_id % 10 == 0:
                vis_path = os.path.join(visualization_dir, f"frame_{frame_id:06d}.jpg")
                cv2.imwrite(vis_path, vis_frame)
        
        # Progress update
        if frame_id % 30 == 0:
            progress = (frame_id / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_id}/{total_frames} frames)")
    
    cap.release()
    
    # Write MOT format file
    print(f"Writing MOT format file: {mot_file}")
    with open(mot_file, 'w') as f:
        for result in tracking_results:
            x, y, w, h = result['bbox']
            # MOT format: frame_id, object_id, x, y, width, height, confidence, x_3d, y_3d, z_3d
            f.write(f"{result['frame_id']},{result['object_id']},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                   f"{result['confidence']:.2f},-1,-1,-1\n")
    
    print(f"\n✓ Processing complete!")
    print(f"  - Tracking results: {mot_file}")
    print(f"  - Visualizations: {visualization_dir}")
    print(f"  - Total objects tracked: {len(set(r['object_id'] for r in tracking_results))}")
    print(f"\nNext step: Review and correct the annotations using annotate_gt.py")
    
    return mot_file, visualization_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video with BOTSORT and export MOT format')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for annotations (default: same as video)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    try:
        process_video(args.video_path, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)