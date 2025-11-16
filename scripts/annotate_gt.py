"""
Simple annotation tool for correcting BOTSORT tracking results.
Allows manual correction of track IDs, adding/removing boxes, and exporting corrected ground truth.
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class AnnotationTool:
    def __init__(self, video_path, mot_file, output_file):
        self.video_path = video_path
        self.mot_file = mot_file
        self.output_file = output_file
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load MOT annotations - check if saved progress exists first
        if os.path.exists(output_file):
            print(f"✓ Found saved progress: {output_file}")
            print("Loading saved annotations...")
            self.annotations = self.load_mot_file(output_file)
            print(f"Resuming from saved progress ({len(self.annotations)} frames annotated)")
        else:
            print(f"Starting fresh from: {mot_file}")
            self.annotations = self.load_mot_file(mot_file)
        
        # Current frame - resume from last position if saved
        self.current_frame_id = self.get_last_frame_id() if os.path.exists(output_file) else 1
        self.frame = None
        
        # Drawing state
        self.drawing = False
        self.start_point = None
        self.selected_box_idx = None
        
        # Colors for different track IDs
        self.colors = {}
        
        print(f"Loaded {len(self.annotations)} annotations across {self.total_frames} frames")
        print(f"Starting at frame: {self.current_frame_id}")
        print("\nControls:")
        print("  N/Right Arrow: Next frame")
        print("  P/Left Arrow: Previous frame")
        print("  Click box: Select box")
        print("  I: Change ID of selected box")
        print("  D: Delete selected box")
        print("  A: Add new box (draw rectangle)")
        print("  S: Save progress (can continue later)")
        print("  Q: Quit (auto-saves)")
    
    def load_mot_file(self, mot_file):
        """Load MOT format annotations into dictionary"""
        annotations = defaultdict(list)
        
        if not os.path.exists(mot_file):
            print(f"Warning: MOT file not found: {mot_file}")
            return annotations
        
        with open(mot_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) >= 6:
                    frame_id = int(float(parts[0]))
                    object_id = int(float(parts[1]))
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    
                    annotations[frame_id].append({
                        'object_id': object_id,
                        'bbox': [x, y, w, h],
                        'frame_id': frame_id
                    })
        
        return annotations
    
    def get_color(self, track_id):
        """Get color for track ID"""
        if track_id not in self.colors:
            # Generate distinct colors
            hue = (track_id * 137.508) % 360  # Golden angle for color distribution
            self.colors[track_id] = self.hsv_to_bgr(hue, 0.7, 0.9)
        return self.colors[track_id]
    
    def hsv_to_bgr(self, h, s, v):
        """Convert HSV to BGR"""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h/360, s, v)
        return (int(b*255), int(g*255), int(r*255))
    
    def load_frame(self, frame_id):
        """Load frame from video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame.copy()
            self.current_frame_id = frame_id
        return ret
    
    def draw_annotations(self, frame):
        """Draw annotations on frame"""
        if self.current_frame_id not in self.annotations:
            return frame
        
        frame_copy = frame.copy()
        boxes = self.annotations[self.current_frame_id]
        
        for idx, box_data in enumerate(boxes):
            x, y, w, h = box_data['bbox']
            track_id = box_data['object_id']
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            color = self.get_color(track_id)
            thickness = 3 if idx == self.selected_box_idx else 2
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw ID label
            label = f"ID:{track_id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_copy, (x1, y1 - text_height - 5), 
                         (x1 + text_width, y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw selection indicator
        if self.selected_box_idx is not None and self.selected_box_idx < len(boxes):
            box_data = boxes[self.selected_box_idx]
            x, y, w, h = box_data['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cv2.rectangle(frame_copy, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 0), 2)
        
        # Draw frame info
        info_text = f"Frame: {self.current_frame_id}/{self.total_frames} | Objects: {len(boxes)}"
        cv2.putText(frame_copy, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_copy
    
    def find_box_at_point(self, x, y):
        """Find box index at point (x, y)"""
        if self.current_frame_id not in self.annotations:
            return None
        
        boxes = self.annotations[self.current_frame_id]
        for idx, box_data in enumerate(boxes):
            bx, by, bw, bh = box_data['bbox']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return idx
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing:
                self.start_point = (x, y)
            else:
                # Select box
                idx = self.find_box_at_point(x, y)
                self.selected_box_idx = idx
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                # Add new box
                x1, y1 = self.start_point
                x2, y2 = x, y
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Get next available ID
                max_id = 0
                for frame_id, boxes in self.annotations.items():
                    for box in boxes:
                        max_id = max(max_id, box['object_id'])
                
                new_box = {
                    'object_id': max_id + 1,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'frame_id': self.current_frame_id
                }
                
                if self.current_frame_id not in self.annotations:
                    self.annotations[self.current_frame_id] = []
                self.annotations[self.current_frame_id].append(new_box)
                
                self.drawing = False
                self.start_point = None
    
    def run(self):
        """Run annotation tool"""
        window_name = "Ground Truth Annotation Tool"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        if not self.load_frame(self.current_frame_id):
            print("Error: Could not load frame")
            return
        
        while True:
            display_frame = self.draw_annotations(self.frame)
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                # Auto-save on quit
                print("\nAuto-saving before quit...")
                self.save_annotations()
                break
            elif key == ord('n') or key == 83:  # N or Right Arrow
                if self.current_frame_id < self.total_frames:
                    self.load_frame(self.current_frame_id + 1)
                    self.selected_box_idx = None
            elif key == ord('p') or key == 81:  # P or Left Arrow
                if self.current_frame_id > 1:
                    self.load_frame(self.current_frame_id - 1)
                    self.selected_box_idx = None
            elif key == ord('i'):  # Change ID
                if (self.selected_box_idx is not None and 
                    self.current_frame_id in self.annotations):
                    boxes = self.annotations[self.current_frame_id]
                    if self.selected_box_idx < len(boxes):
                        try:
                            new_id = int(input(f"Enter new ID (current: {boxes[self.selected_box_idx]['object_id']}): "))
                            boxes[self.selected_box_idx]['object_id'] = new_id
                        except ValueError:
                            print("Invalid ID")
            elif key == ord('d'):  # Delete box
                if (self.selected_box_idx is not None and 
                    self.current_frame_id in self.annotations):
                    boxes = self.annotations[self.current_frame_id]
                    if self.selected_box_idx < len(boxes):
                        boxes.pop(self.selected_box_idx)
                        self.selected_box_idx = None
            elif key == ord('a'):  # Add box mode
                self.drawing = True
                print("Click and drag to draw a box")
            elif key == ord('s'):  # Save
                self.save_annotations()
                print("✓ Progress saved! You can quit and continue later.")
        
        cv2.destroyAllWindows()
        self.cap.release()
    
    def save_annotations(self):
        """Save corrected annotations in MOT format"""
        with open(self.output_file, 'w') as f:
            for frame_id in sorted(self.annotations.keys()):
                for box_data in self.annotations[frame_id]:
                    x, y, w, h = box_data['bbox']
                    f.write(f"{frame_id},{box_data['object_id']},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                           f"1.0,-1,-1,-1\n")
        
        # Save current frame position to a separate file
        state_file = self.output_file.replace('.txt', '_state.txt')
        with open(state_file, 'w') as f:
            f.write(f"last_frame={self.current_frame_id}\n")
        
        print(f"Saved corrected annotations to: {self.output_file}")
    
    def get_last_frame_id(self):
        """Get the last frame position from saved state"""
        state_file = self.output_file.replace('.txt', '_state.txt')
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    for line in f:
                        if line.startswith('last_frame='):
                            return int(line.split('=')[1].strip())
            except:
                pass
        
        # Fallback: find max frame in annotations
        if not os.path.exists(self.output_file):
            return 1
        
        max_frame = 0
        with open(self.output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 1:
                        try:
                            frame_id = int(float(parts[0]))
                            max_frame = max(max_frame, frame_id)
                        except:
                            pass
        
        return max(1, max_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Annotate and correct ground truth')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('mot_file', type=str, help='Path to MOT format file (from process_video_for_gt.py)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output file for corrected annotations (default: adds _corrected to input)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    if not os.path.exists(args.mot_file):
        print(f"Error: MOT file not found: {args.mot_file}")
        sys.exit(1)
    
    if args.output is None:
        base = os.path.splitext(args.mot_file)[0]
        args.output = f"{base}_corrected.txt"
    
    try:
        tool = AnnotationTool(args.video_path, args.mot_file, args.output)
        tool.run()
        tool.save_annotations()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)