"""
Evaluate BOTSORT tracker against ground truth annotations.
Computes MOTA, MOTP, and ID Switch Count.
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import cv2


def load_mot_file(mot_file):
    """Load MOT format annotations"""
    annotations = defaultdict(list)
    
    if not os.path.exists(mot_file):
        raise FileNotFoundError(f"MOT file not found: {mot_file}")
    
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


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
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


def match_detections_to_ground_truth(gt_boxes, det_boxes, iou_threshold=0.5):
    """
    Match detection boxes to ground truth boxes using Hungarian algorithm.
    Returns: matches, unmatched_gt, unmatched_det
    """
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return [], list(range(len(gt_boxes))), list(range(len(det_boxes)))
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(det_boxes), len(gt_boxes)))
    for i, det_box in enumerate(det_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(det_box['bbox'], gt_box['bbox'])
    
    # Use Hungarian algorithm for optimal matching
    from scipy.optimize import linear_sum_assignment
    
    # Maximize IoU (minimize 1 - IoU)
    cost_matrix = 1 - iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    matched_gt = set()
    matched_det = set()
    
    # Filter matches based on IoU threshold
    for row_idx, col_idx in zip(row_indices, col_indices):
        if row_idx < len(det_boxes) and col_idx < len(gt_boxes):
            iou = iou_matrix[row_idx, col_idx]
            if iou >= iou_threshold:
                matches.append((row_idx, col_idx, iou))
                matched_gt.add(col_idx)
                matched_det.add(row_idx)
    
    # Find unmatched
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_det = [i for i in range(len(det_boxes)) if i not in matched_det]
    
    return matches, unmatched_gt, unmatched_det


def track_id_consistency_check(gt_annotations, det_annotations, iou_threshold=0.5):
    """
    Check ID consistency across frames and count ID switches.
    Returns: id_switches, track_mapping
    """
    # Build track histories
    gt_tracks = defaultdict(list)  # gt_id -> [(frame_id, bbox_idx), ...]
    det_tracks = defaultdict(list)  # det_id -> [(frame_id, bbox_idx), ...]
    
    for frame_id in sorted(gt_annotations.keys()):
        for idx, gt_box in enumerate(gt_annotations[frame_id]):
            gt_tracks[gt_box['object_id']].append((frame_id, idx))
    
    for frame_id in sorted(det_annotations.keys()):
        for idx, det_box in enumerate(det_annotations[frame_id]):
            det_tracks[det_box['object_id']].append((frame_id, idx))
    
    # Match tracks across frames
    id_switches = 0
    track_mapping = {}  # gt_id -> det_id mapping
    
    # Process frames in order
    for frame_id in sorted(set(list(gt_annotations.keys()) + list(det_annotations.keys()))):
        if frame_id not in gt_annotations or frame_id not in det_annotations:
            continue
        
        gt_boxes = gt_annotations[frame_id]
        det_boxes = det_annotations[frame_id]
        
        # Match boxes in this frame
        matches, _, _ = match_detections_to_ground_truth(gt_boxes, det_boxes, iou_threshold)
        
        # Check for ID switches
        for det_idx, gt_idx, iou in matches:
            gt_id = gt_boxes[gt_idx]['object_id']
            det_id = det_boxes[det_idx]['object_id']
            
            if gt_id in track_mapping:
                # Check if ID changed (ID switch)
                if track_mapping[gt_id] != det_id:
                    id_switches += 1
                    # Update mapping
                    track_mapping[gt_id] = det_id
            else:
                # New mapping
                track_mapping[gt_id] = det_id
    
    return id_switches, track_mapping


def evaluate_tracker(gt_file, det_file, iou_threshold=0.5):
    """
    Evaluate tracker against ground truth.
    Returns: MOTA, MOTP, ID Switch Count, and other metrics
    """
    print(f"Loading ground truth: {gt_file}")
    gt_annotations = load_mot_file(gt_file)
    
    print(f"Loading detection results: {det_file}")
    det_annotations = load_mot_file(det_file)
    
    # Statistics
    total_gt = 0
    total_det = 0
    total_matches = 0
    total_iou = 0.0
    false_positives = 0
    false_negatives = 0
    
    # Process each frame
    all_frame_ids = sorted(set(list(gt_annotations.keys()) + list(det_annotations.keys())))
    
    print(f"Evaluating {len(all_frame_ids)} frames...")
    
    for frame_id in all_frame_ids:
        gt_boxes = gt_annotations.get(frame_id, [])
        det_boxes = det_annotations.get(frame_id, [])
        
        total_gt += len(gt_boxes)
        total_det += len(det_boxes)
        
        if len(gt_boxes) > 0 and len(det_boxes) > 0:
            matches, unmatched_gt, unmatched_det = match_detections_to_ground_truth(
                gt_boxes, det_boxes, iou_threshold
            )
            
            total_matches += len(matches)
            false_negatives += len(unmatched_gt)
            false_positives += len(unmatched_det)
            
            # Accumulate IoU for matched pairs
            for det_idx, gt_idx, iou in matches:
                total_iou += iou
        elif len(gt_boxes) > 0:
            false_negatives += len(gt_boxes)
        elif len(det_boxes) > 0:
            false_positives += len(det_boxes)
    
    # Calculate ID switches
    print("Checking ID consistency...")
    id_switches, track_mapping = track_id_consistency_check(gt_annotations, det_annotations, iou_threshold)
    
    # Calculate metrics
    # MOTA = 1 - (FN + FP + IDSW) / GT
    mota = 1.0 - (false_negatives + false_positives + id_switches) / max(total_gt, 1)
    
    # MOTP = Average IoU of matched pairs
    motp = total_iou / max(total_matches, 1)
    
    # Additional metrics
    precision = total_matches / max(total_det, 1)
    recall = total_matches / max(total_gt, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-10)
    
    return {
        'MOTA': mota,
        'MOTP': motp,
        'ID_Switches': id_switches,
        'False_Positives': false_positives,
        'False_Negatives': false_negatives,
        'Total_Matches': total_matches,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'Total_GT_Objects': total_gt,
        'Total_Det_Objects': total_det
    }


def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("TRACKING EVALUATION RESULTS")
    print("="*60)
    print(f"\nMOTA (Multiple Object Tracking Accuracy): {results['MOTA']:.4f} ({results['MOTA']*100:.2f}%)")
    print(f"MOTP (Multiple Object Tracking Precision):  {results['MOTP']:.4f} ({results['MOTP']*100:.2f}%)")
    print(f"ID Switch Count:                            {results['ID_Switches']}")
    print(f"\nDetailed Metrics:")
    print(f"  False Positives:  {results['False_Positives']}")
    print(f"  False Negatives: {results['False_Negatives']}")
    print(f"  Total Matches:   {results['Total_Matches']}")
    print(f"  Precision:      {results['Precision']:.4f} ({results['Precision']*100:.2f}%)")
    print(f"  Recall:          {results['Recall']:.4f} ({results['Recall']*100:.2f}%)")
    print(f"  F1 Score:        {results['F1_Score']:.4f}")
    print(f"\nTotals:")
    print(f"  Ground Truth Objects: {results['Total_GT_Objects']}")
    print(f"  Detected Objects:    {results['Total_Det_Objects']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate BOTSORT tracker against ground truth')
    parser.add_argument('gt_file', type=str, help='Path to ground truth MOT file')
    parser.add_argument('det_file', type=str, help='Path to detection results MOT file')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file not found: {args.gt_file}")
        sys.exit(1)
    
    if not os.path.exists(args.det_file):
        print(f"Error: Detection file not found: {args.det_file}")
        sys.exit(1)
    
    try:
        results = evaluate_tracker(args.gt_file, args.det_file, args.iou_threshold)
        print_results(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write("TRACKING EVALUATION RESULTS\n")
                f.write("="*60 + "\n\n")
                for key, value in results.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            print(f"Results saved to: {args.output}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)