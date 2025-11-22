"""
Test script for YOLOv8 (person detection) + best.pt (dress code) and BoTSORT.
Supports: HEIC, MOV, MP4, PNG, JPG/JPEG

This script performs two-stage detection:
1. Detect persons using YOLOv8n
2. Track persons using BoTSORT
3. Detect dress code classes using best.pt model
4. Display detected classes with bounding boxes and confidence scores

Usage:
    python test_yolov8_botsort.py <input_file>
    python test_yolov8_botsort.py <input_file> --output <output_path> --conf 0.5
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from src.botsort_tracker import BotSORT

# Try to import HEIC support
try:
    from pillow_heif import register_heif_opener
    from PIL import Image
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    try:
        import pyheif
        from PIL import Image
        HEIC_SUPPORT = True
    except ImportError:
        HEIC_SUPPORT = False
        print("Warning: HEIC support not available. Install pillow-heif or pyheif for HEIC support.")


def is_image_file(file_path):
    """Check if file is an image based on extension"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.heic', '.heif', '.bmp', '.gif'}
    return Path(file_path).suffix.lower() in image_extensions


def is_video_file(file_path):
    """Check if file is a video based on extension"""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm'}
    return Path(file_path).suffix.lower() in video_extensions


def load_image(file_path):
    """
    Load image file, supporting HEIC format
    Returns numpy array (BGR format for OpenCV)
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in {'.heic', '.heif'}:
        if not HEIC_SUPPORT:
            raise ValueError("HEIC support not available. Install pillow-heif or pyheif.")
        
        try:
            # Try pillow-heif first
            from pillow_heif import register_heif_opener
            from PIL import Image
            register_heif_opener()
            img = Image.open(file_path)
            # Convert PIL image to numpy array (RGB)
            img_array = np.array(img)
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_array
        except Exception as e:
            try:
                # Try pyheif as fallback
                import pyheif
                from PIL import Image
                heif_file = pyheif.read(file_path)
                img = Image.frombytes(
                    heif_file.mode,
                    heif_file.size,
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_array
            except Exception as e2:
                raise ValueError(f"Failed to load HEIC file: {e2}")
    else:
        # Standard image formats
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Failed to load image: {file_path}")
        return img


def format_dress_class(class_name):
    """Format dress class names for better readability"""
    class_mapping = {
        'polo_shirt': 'Polo Shirt',
        'pants': 'Pants',
        'shoes': 'Shoes',
        'blouse': 'Blouse',
        'skirt': 'Skirt',
        'doll_shoes': 'Doll Shoes'
    }
    return class_mapping.get(class_name, class_name.replace('_', ' ').title())


def detect_dress_code(person_crop, dress_model, crop_offset=(0, 0), conf_threshold=0.5):
    """Detect dress code items for a person crop using best.pt model"""
    try:
        # Run dress code detection on person crop
        results = dress_model(person_crop, conf=conf_threshold)
        
        dress_items = []
        crop_x_offset, crop_y_offset = crop_offset
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Get class name from model
                    class_name = dress_model.names[class_id]
                    # Get bbox coordinates (relative to crop)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Convert to frame coordinates
                    dress_items.append({
                        'class': format_dress_class(class_name),
                        'class_raw': class_name,  # Keep raw class name
                        'confidence': round(confidence, 2),
                        'bbox': [x1 + crop_x_offset, y1 + crop_y_offset, 
                                x2 + crop_x_offset, y2 + crop_y_offset]  # Store bbox in frame coordinates
                    })
        
        # Sort by confidence (highest first)
        dress_items.sort(key=lambda x: x['confidence'], reverse=True)
        
        return dress_items
        
    except Exception as e:
        print(f"Error in dress code detection: {e}")
        return []


def detect_persons_with_dress(frame, person_model, dress_model, tracker, conf_threshold=0.5):
    """Two-stage detection: first detect persons, then detect dress code"""
    try:
        # Stage 1: Detect persons using YOLOv8n
        results = person_model(frame, conf=0.5)
        
        # Process person detections
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if it's a person (class 0 in COCO dataset)
                    if class_id == 0 and confidence > 0.5:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': round(confidence, 2),
                            'class': 'person'
                        })
        
        # Update tracker with detections FIRST to get track IDs
        if detections:
            # Convert detections to tracker format [x, y, w, h, confidence]
            dets = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                w, h = x2 - x1, y2 - y1
                dets.append([x1, y1, w, h, det['confidence']])
            
            dets = np.array(dets)
            
            # Update tracker
            tracked_objects = tracker.update(dets, frame)
            
            # Add tracking IDs to detections
            for i, track in enumerate(tracked_objects):
                if i < len(detections):
                    detections[i]['track_id'] = int(track[4])
                    # Convert back to [x1, y1, x2, y2] format
                    x, y, w, h = track[0], track[1], track[2], track[3]
                    detections[i]['bbox'] = [int(x), int(y), int(x + w), int(y + h)]
        
        # Stage 2: Detect dress code for each person
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Extract person crop with some padding
            padding = 10
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(frame.shape[1], x2 + padding)
            crop_y2 = min(frame.shape[0], y2 + padding)
            
            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if person_crop.size == 0:
                detection['dress_items'] = []
                continue
            
            # Detect dress code items for this person
            dress_items = detect_dress_code(person_crop, dress_model, crop_offset=(crop_x1, crop_y1), conf_threshold=conf_threshold)
            detection['dress_items'] = dress_items
            
            # Create summary of detected classes
            if dress_items:
                detected_classes = [item['class'] for item in dress_items]
                detection['dress_summary'] = f"Detected: {', '.join(detected_classes)}"
            else:
                detection['dress_summary'] = "No dress items detected"
        
        return detections
        
    except Exception as e:
        print(f"Error in two-stage detection: {e}")
        import traceback
        traceback.print_exc()
        return []


def draw_detections_frame(frame, detections, dress_model):
    """Draw bounding boxes on a video frame with tracking IDs and detected dress code classes"""
    try:
        # First, draw dress code item bounding boxes directly on the frame (blue boxes)
        for detection in detections:
            dress_items = detection.get('dress_items', [])
            
            # Draw dress code item bounding boxes (bboxes are already in frame coordinates)
            for item in dress_items:
                if 'bbox' in item:
                    bx1, by1, bx2, by2 = item['bbox']
                    bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
                    
                    # Draw blue box for dress items
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                    
                    # Label with class name and confidence
                    class_name = item['class']
                    conf = item['confidence']
                    label_text = f"{class_name} {conf*100:.0f}%"
                    label_scale = 0.4
                    label_thickness = 1
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)
                    pad = 3
                    
                    # Draw filled background above the box if room, otherwise inside
                    rect_top = by1 - th - pad*2 if by1 - th - pad*2 > 0 else by1 + pad
                    rect_bottom = rect_top + th + pad*2
                    cv2.rectangle(frame, (bx1, rect_top), (bx1 + tw + pad*2, rect_bottom), (255, 0, 0), -1)
                    cv2.putText(frame, label_text, (bx1 + pad, rect_bottom - pad), 
                               cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), label_thickness)
        
        # Draw person bounding boxes with detected classes summary
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            track_id = detection.get('track_id', 'N/A')
            dress_summary = detection.get('dress_summary', 'No dress items detected')
            dress_items = detection.get('dress_items', [])
            
            # Use track ID based color for person boxes
            color_int = (track_id * 50) % 255 if track_id != 'N/A' else 128
            color = (color_int, 255, 255 - color_int)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with tracking ID and person confidence
            label1 = f"ID:{track_id} Person: {confidence:.2f}"
            
            # Draw first label (person info)
            label_size1 = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size1[1] - 10), 
                         (x1 + label_size1[0], y1), color, -1)
            cv2.putText(frame, label1, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw detected classes summary
            current_y = y1 - label_size1[1] - 15
            summary_text = dress_summary
            text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw background for summary
            cv2.rectangle(frame, (x1, current_y - text_size[1] - 5), 
                         (x1 + text_size[0] + 5, current_y + 5), color, -1)
            cv2.putText(frame, summary_text, (x1 + 2, current_y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    except Exception as e:
        print(f"Error drawing detections on frame: {e}")
        import traceback
        traceback.print_exc()
        return frame


def process_image(image_path, output_path=None, person_model=None, dress_model=None, tracker=None, conf_threshold=0.5):
    """Process a single image with YOLOv8, best.pt, and BoTSORT"""
    print(f"\n{'='*60}")
    print(f"Processing Image: {image_path}")
    print(f"{'='*60}")
    
    # Load image
    try:
        print("Loading image...")
        image = load_image(image_path)
        print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    except Exception as e:
        print(f"Error loading image: {e}")
        return False
    
    # Two-stage detection: persons + dress code
    print("Detecting persons with YOLOv8 and dress code with best.pt...")
    detections = detect_persons_with_dress(image, person_model, dress_model, tracker, conf_threshold=conf_threshold)
    print(f"Found {len(detections)} person(s)")
    
    if len(detections) == 0:
        print("No persons detected in the image.")
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Original image saved to: {output_path}")
        return True
    
    # Create visualization
    vis_image = image.copy()
    vis_image = draw_detections_frame(vis_image, detections, dress_model)
    
    # Save output
    if output_path is None:
        base_name = Path(image_path).stem
        output_dir = Path(image_path).parent / "test_results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{base_name}_result.jpg"
    
    cv2.imwrite(str(output_path), vis_image)
    print(f"\n✓ Results saved to: {output_path}")
    print(f"  - Persons detected: {len(detections)}")
    
    # Print detection summary
    for det in detections:
        dress_items = det.get('dress_items', [])
        if dress_items:
            classes = [item['class'] for item in dress_items]
            print(f"  - Track ID {det.get('track_id', 'N/A')}: {', '.join(classes)}")
        else:
            print(f"  - Track ID {det.get('track_id', 'N/A')}: No dress items detected")
    
    return True


def process_video(video_path, output_path=None, person_model=None, dress_model=None, tracker=None, conf_threshold=0.5):
    """Process video with YOLOv8, best.pt, and BoTSORT"""
    print(f"\n{'='*60}")
    print(f"Processing Video: {video_path}")
    print(f"{'='*60}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output video writer
    if output_path is None:
        base_name = Path(video_path).stem
        output_dir = Path(video_path).parent / "test_results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{base_name}_result.mp4"
    
    # Try different codecs for better compatibility
    codecs = ['mp4v', 'XVID', 'avc1', 'H264']
    out = None
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Using codec: {codec}")
            break
        else:
            out = None
    
    if out is None or not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        print("Tried codecs:", codecs)
        cap.release()
        return False
    
    # Process video frame by frame
    frame_id = 0
    total_detections = 0
    unique_tracks = set()
    
    print("\nProcessing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # Two-stage detection: persons + dress code
        detections = detect_persons_with_dress(frame, person_model, dress_model, tracker, conf_threshold=conf_threshold)
        
        if len(detections) > 0:
            total_detections += len(detections)
            for det in detections:
                unique_tracks.add(det.get('track_id', 0))
        
        # Draw detections on frame
        frame = draw_detections_frame(frame, detections, dress_model)
        
        # Write frame to output video
        out.write(frame)
        
        # Progress update
        if frame_id % 30 == 0 or frame_id == total_frames:
            progress = (frame_id / total_frames) * 100 if total_frames > 0 else 0
            print(f"Progress: {progress:.1f}% ({frame_id}/{total_frames} frames)")
    
    cap.release()
    out.release()
    
    print(f"\n✓ Video processing complete!")
    print(f"  - Output saved to: {output_path}")
    print(f"  - Total frames processed: {frame_id}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Unique tracks: {len(unique_tracks)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Test YOLOv8 (person) + best.pt (dress code) + BoTSORT with images and videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  Images: PNG, JPG, JPEG, HEIC, HEIF, BMP, GIF
  Videos: MP4, MOV, AVI, MKV, FLV, WMV, WEBM

Examples:
  python test_yolov8_botsort.py image.jpg
  python test_yolov8_botsort.py video.mp4 --output result.mp4
  python test_yolov8_botsort.py image.heic
  python test_yolov8_botsort.py video.mov --conf 0.6
        """
    )
    parser.add_argument('input_file', type=str, help='Path to input image or video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (default: auto-generated in test_results folder)')
    parser.add_argument('--person-model', type=str, default='models/yolov8n.pt',
                       help='Path to YOLOv8 person detection model (default: models/yolov8n.pt)')
    parser.add_argument('--dress-model', type=str, default='models/best.pt',
                       help='Path to dress code detection model (default: models/best.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                       help='Confidence threshold for detection (default: 0.5)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Check if model files exist
    if not os.path.exists(args.person_model):
        print(f"Error: Person model file not found: {args.person_model}")
        sys.exit(1)
    
    if not os.path.exists(args.dress_model):
        print(f"Error: Dress model file not found: {args.dress_model}")
        sys.exit(1)
    
    # Load YOLOv8 person detection model
    print(f"Loading YOLOv8 person detection model from: {args.person_model}")
    try:
        person_model = YOLO(args.person_model)
        print("✓ Person model loaded successfully")
    except Exception as e:
        print(f"Error loading person model: {e}")
        sys.exit(1)
    
    # Load dress code detection model
    print(f"Loading dress code detection model from: {args.dress_model}")
    try:
        dress_model = YOLO(args.dress_model)
        print("✓ Dress code model loaded successfully")
        print(f"  - Classes: {list(dress_model.names.values())}")
    except Exception as e:
        print(f"Error loading dress model: {e}")
        sys.exit(1)
    
    # Initialize BoTSORT tracker
    print("Initializing BoTSORT tracker...")
    tracker = BotSORT()
    print("✓ Tracker initialized")
    
    # Determine file type and process
    if is_image_file(args.input_file):
        success = process_image(args.input_file, args.output, person_model, dress_model, tracker, args.conf)
    elif is_video_file(args.input_file):
        success = process_video(args.input_file, args.output, person_model, dress_model, tracker, args.conf)
    else:
        print(f"Error: Unsupported file format: {Path(args.input_file).suffix}")
        print("Supported formats:")
        print("  Images: PNG, JPG, JPEG, HEIC, HEIF, BMP, GIF")
        print("  Videos: MP4, MOV, AVI, MKV, FLV, WMV, WEBM")
        sys.exit(1)
    
    if success:
        print("\n" + "="*60)
        print("Processing completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Processing failed!")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
