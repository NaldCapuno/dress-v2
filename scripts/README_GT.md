# Ground Truth Generation and Evaluation Guide

This guide explains how to generate ground truth annotations and evaluate your BOTSORT tracker.

## Overview

1. **Process Video**: Generate initial annotations using BOTSORT
2. **Annotate**: Manually correct the annotations
3. **Evaluate**: Compare tracker results against ground truth

## Step 1: Process Video with BOTSORT

Process your test video to generate initial tracking annotations:

python scripts/process_video_for_gt.py path/to/your/video.mp4 
This will:
- Run BOTSORT tracker on the video
- Export tracking results in MOT format
- Create visualization frames

Output files:
- `gt_annotations/{video_name}_botsort.txt` - MOT format tracking results
- `gt_annotations/{video_name}_visualization/` - Visualization frames

## Step 2: Manual Annotation and Correction

Review and correct the BOTSORT annotations:

python scripts/annotate_gt.py path/to/video.mp4 gt_annotations/{video_name}_botsort.txt**Controls:**
- `N` or `Right Arrow`: Next frame
- `P` or `Left Arrow`: Previous frame
- `Click box`: Select a bounding box
- `I`: Change ID of selected box
- `D`: Delete selected box
- `A`: Add new box (click and drag)
- `S`: Save corrected annotations
- `Q`: Quit

**Important**: 
- Fix ID switches (when same person gets different IDs)
- Add missed detections
- Remove false positives
- Adjust bounding boxes if needed

The corrected annotations will be saved as `{video_name}_botsort_corrected.txt`

## Step 3: Evaluate Tracker

Compare BOTSORT results against corrected ground truth:
ash
python scripts/evaluate_tracker.py gt_annotations/{video_name}_botsort_corrected.txt gt_annotations/{video_name}_botsort.txtThis will compute:
- **MOTA** (Multiple Object Tracking Accuracy)
- **MOTP** (Multiple Object Tracking Precision)
- **ID Switch Count**
- Additional metrics (Precision, Recall, F1 Score)

## MOT Format

All annotation files use MOT Challenge format: