"""
Evaluate BOTSORT tracker against ground truth annotations using motmetrics.
Computes standard MOT Challenge metrics: MOTA, MOTP, IDF1, IDSW, FP, FN,
Precision, Recall, and F1-score.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import motmetrics as mm


def calculate_iou_matrix(gt_boxes, det_boxes, iou_threshold=0.5):
    """
    Calculate IoU distance matrix (1 - IoU).
    Pairs with IoU < threshold get distance = inf (cannot be matched).
    """
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return np.array([])

    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    det_boxes = np.asarray(det_boxes, dtype=np.float64)

    iou_matrix = np.zeros((len(gt_boxes), len(det_boxes)), dtype=np.float64)

    for i, gt in enumerate(gt_boxes):
        x1g, y1g, wg, hg = gt
        x2g, y2g = x1g + wg, y1g + hg
        area_g = wg * hg

        if area_g <= 0:
            continue

        for j, dt in enumerate(det_boxes):
            x1d, y1d, wd, hd = dt
            x2d, y2d = x1d + wd, y1d + hd
            area_d = wd * hd

            if area_d <= 0:
                iou_matrix[i, j] = np.inf
                continue

            # Intersection
            xi1, yi1 = max(x1g, x1d), max(y1g, y1d)
            xi2, yi2 = min(x2g, x2d), min(y2g, y2d)

            if xi2 > xi1 and yi2 > yi1:
                inter = (xi2 - xi1) * (yi2 - yi1)
                union = area_g + area_d - inter
                iou = inter / union if union > 0 else 0
            else:
                iou = 0.0

            if iou < iou_threshold:
                iou_matrix[i, j] = np.inf
            else:
                iou_matrix[i, j] = 1 - iou

    return iou_matrix


def load_mot_file(path):
    """
    Loads MOT-format file as a DataFrame with:
    [FrameId, Id, X, Y, Width, Height]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = []

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 6:
                print(f"Skipping line {line_num}: insufficient columns")
                continue

            try:
                frame = int(float(parts[0]))
                oid = int(float(parts[1]))
                x, y = float(parts[2]), float(parts[3])
                w, h = float(parts[4]), float(parts[5])

                if w <= 0 or h <= 0:
                    print(f"Skipping invalid bbox in line {line_num}")
                    continue

                data.append({
                    "FrameId": frame,
                    "Id": oid,
                    "X": x,
                    "Y": y,
                    "Width": w,
                    "Height": h
                })
            except:
                print(f"Warning: error parsing line {line_num}")
                continue

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} entries from: {path}")
    return df


def evaluate_tracker(gt_file, det_file, iou_threshold=0.5):
    """ Main evaluation logic using motmetrics. """

    gt_df = load_mot_file(gt_file)
    det_df = load_mot_file(det_file)

    acc = mm.MOTAccumulator(auto_id=False)

    all_frames = sorted(set(gt_df.FrameId.unique()) |
                        set(det_df.FrameId.unique()))

    print(f"Evaluating {len(all_frames)} frames...")

    for frame in all_frames:
        gt_frame = gt_df[gt_df.FrameId == frame]
        det_frame = det_df[det_df.FrameId == frame]

        gt_boxes = [[r.X, r.Y, r.Width, r.Height] for _, r in gt_frame.iterrows()]
        det_boxes = [[r.X, r.Y, r.Width, r.Height] for _, r in det_frame.iterrows()]

        gt_ids = list(gt_frame.Id.values)
        det_ids = list(det_frame.Id.values)

        if gt_boxes and det_boxes:
            dist = calculate_iou_matrix(gt_boxes, det_boxes, iou_threshold)
            acc.update(gt_ids, det_ids, dist, frameid=frame)
        else:
            acc.update(gt_ids, det_ids, np.array([]), frameid=frame)

    print("Computing MOT metrics...")
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="BOTSORT")

    # Convert summary to dictionary, handling NaN values
    results = {}
    for metric in summary.columns:
        value = summary[metric].iloc[0]
        if pd.isna(value):
            results[metric] = 0.0
        elif isinstance(value, (int, float, np.integer, np.floating)):
            results[metric] = float(value)
        else:
            # Try to convert to float if possible
            try:
                results[metric] = float(value)
            except (ValueError, TypeError):
                results[metric] = str(value)

    # ----------------------------------------------------
    # ADD CORRECT PRECISION / RECALL / F1 FORMULAS
    # ----------------------------------------------------
    # Calculate total GT objects from the dataframe (more reliable)
    total_gt_objects = len(gt_df)
    total_det_objects = len(det_df)
    
    # Get metrics from motmetrics
    FN = float(results.get("num_misses", 0))
    FP = float(results.get("num_false_positives", 0))
    
    # True Positives = total GT objects - false negatives (matched objects)
    TP = total_gt_objects - FN
    
    # Ensure TP is not negative (shouldn't happen, but safety check)
    if TP < 0:
        print(f"Warning: TP calculated as negative ({TP}). Using motmetrics num_objects if available.")
        num_objects_mm = float(results.get("num_objects", 0))
        if num_objects_mm > 0:
            TP = num_objects_mm - FN
        else:
            TP = 0
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = f1
    
    # Store actual counts
    results["Total_GT_Objects"] = total_gt_objects
    results["Total_Det_Objects"] = total_det_objects
    results["True_Positives"] = TP
    results["IoU_Threshold"] = iou_threshold
    results["Num_Frames"] = len(all_frames)

    return results, summary


def print_results(results, summary):
    """ Pretty output formatting """

    print("\n" + "="*70)
    print("MOT TRACKING EVALUATION RESULTS")
    print("="*70)

    print(f"\nCore MOT Metrics:")
    # Ensure all values are numeric
    mota = float(results.get('mota', 0))
    motp = float(results.get('motp', 0))
    idf1 = float(results.get('idf1', 0))
    
    # MOTP is average distance (1 - IoU), convert to IoU for clarity
    avg_iou = 1.0 - motp if motp <= 1.0 else 0.0
    
    print(f"  MOTA:          {mota:.4f} ({mota*100:.2f}%)")
    print(f"  MOTP (distance): {motp:.4f} (avg IoU: {avg_iou:.4f} = {avg_iou*100:.2f}%)")
    print(f"  IDF1:          {idf1:.4f} ({idf1*100:.2f}%)")

    print("\nCounts:")
    # Use our calculated values instead of motmetrics num_objects
    total_gt = int(results.get('Total_GT_Objects', 0))
    total_det = int(results.get('Total_Det_Objects', 0))
    TP = int(results.get('True_Positives', 0))
    FN = float(results.get('num_misses', 0))
    FP = float(results.get('num_false_positives', 0))
    IDSW = float(results.get('num_switches', 0))
    
    print(f"  Total GT Objects:   {total_gt}")
    print(f"  Total Det Objects:  {total_det}")
    print(f"  True Positives (TP): {TP}")
    print(f"  False Positives (FP): {int(FP)}")
    print(f"  False Negatives (FN): {int(FN)}")
    print(f"  ID Switches:         {int(IDSW)}")

    print("\nClassification Metrics:")
    # Ensure all values are numeric
    precision = float(results.get('precision', 0))
    recall = float(results.get('recall', 0))
    f1 = float(results.get('f1', 0))
    print(f"  Precision:     {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:        {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:      {f1:.4f} ({f1*100:.2f}%)")

    print("\nAdditional Info:")
    print(f"  Frames Evaluated: {int(results.get('Num_Frames', 0))}")
    print(f"  IoU Threshold:    {float(results.get('IoU_Threshold', 0.5))}")
    
    # Show track quality metrics if available
    # Ensure all values are numeric
    mt = float(results.get('mostly_tracked', 0))
    ml = float(results.get('mostly_lost', 0))
    pt = float(results.get('partially_tracked', 0))
    frag = float(results.get('frag', 0))
    
    if mt > 0 or ml > 0 or pt > 0 or frag > 0:
        print("\nTrack Quality Metrics:")
        print(f"  Mostly Tracked (MT):   {int(mt)}")
        print(f"  Mostly Lost (ML):      {int(ml)}")
        print(f"  Partially Tracked (PT): {int(pt)}")
        print(f"  Fragmentation (Frag):   {int(frag)}")

    print("\n" + "="*70)
    print("Full Metrics Summary (motmetrics):")
    print(summary.to_string())
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BOTSORT Tracker with MOTMetrics")
    parser.add_argument("gt_file", type=str, help="Ground truth MOT file")
    parser.add_argument("det_file", type=str, help="Detection MOT file")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for matching")
    args = parser.parse_args()

    results, summary = evaluate_tracker(args.gt_file, args.det_file, args.iou_threshold)
    print_results(results, summary)
