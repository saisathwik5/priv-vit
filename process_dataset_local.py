#!/usr/bin/env python3
"""
Local Dataset Processor for EPIC-KITCHENS-100
==============================================

Processes raw RGB frame tarballs into 16-frame .npz clips ready for
upload to Google Drive. Output matches the folder hierarchy expected
by priv_vit_train.ipynb on Colab.

Usage:
    python process_dataset_local.py

Output structure (upload cv_project/ to Google Drive):
    cv_project/
    ├── annotations/
    │   ├── EPIC_100_train.csv
    │   └── EPIC_100_validation.csv
    ├── processed_clips/
    │   ├── train/
    │   │   ├── P01_101_0.npz
    │   │   ├── ...
    │   │   └── .processing_complete
    │   └── val/
    │       ├── ...
    │       └── .processing_complete
    ├── checkpoints/
    └── logs/

Requirements:
    pip install opencv-python-headless numpy pandas tqdm
"""

import os
import sys
import tarfile
import io
import subprocess
import glob
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the downloaded tarballs live
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')

# Output root — this folder gets uploaded to Google Drive as-is
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'cv_project')

PARTICIPANTS = ['P01', 'P06', 'P07', 'P09', 'P11', 'P12']
NUM_FRAMES = 16
IMG_SIZE = 224
NUM_WORKERS = 4  # Parallel tar extraction workers
VAL_SPLIT_RATIO = 0.2  # 20% of available videos go to validation
RANDOM_SEED = 42

# ============================================================
# Step 1: Get annotations and create custom train/val split
# ============================================================
def setup_annotations(available_video_ids):
    """Download annotations and create custom train/val split.

    The official EPIC-KITCHENS-100 validation set uses original EPIC-55
    videos (P0X_0XX IDs), which we don't have. We only have extension
    videos (P0X_1XX). So we create a custom 80/20 split by video from
    the available data.
    """
    ann_dir = os.path.join(OUTPUT_ROOT, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)

    train_csv = os.path.join(ann_dir, 'EPIC_100_train.csv')
    val_csv = os.path.join(ann_dir, 'EPIC_100_validation.csv')

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        print("✅ Annotations already present.")
    else:
        print("📥 Downloading annotations from GitHub...")
        ANNOTATIONS_REPO = 'https://github.com/epic-kitchens/epic-kitchens-100-annotations.git'
        subprocess.run(['git', 'clone', ANNOTATIONS_REPO, ann_dir],
                       check=True, capture_output=True)
        print("✅ Annotations downloaded.")

    # Load both official splits and merge
    official_train = pd.read_csv(train_csv)
    official_val = pd.read_csv(val_csv)
    all_segments = pd.concat([official_train, official_val], ignore_index=True)
    print(f"   Total annotation segments: {len(all_segments)}")

    # Filter to our participants AND available videos only
    available_pids = set(PARTICIPANTS)
    available_vids = set(available_video_ids)
    filtered = all_segments[
        (all_segments['participant_id'].isin(available_pids)) &
        (all_segments['video_id'].isin(available_vids))
    ].copy()
    print(f"   Segments with available videos: {len(filtered)}")
    print(f"   Unique videos: {filtered['video_id'].nunique()}")

    # Custom split: hold out ~20% of videos per participant for validation
    np.random.seed(RANDOM_SEED)
    train_vids = []
    val_vids = []
    for pid in sorted(available_pids):
        pid_videos = sorted(filtered[filtered['participant_id'] == pid]['video_id'].unique())
        n_val = max(1, int(len(pid_videos) * VAL_SPLIT_RATIO))
        shuffled = np.random.permutation(pid_videos)
        val_vids.extend(shuffled[:n_val])
        train_vids.extend(shuffled[n_val:])

    val_vid_set = set(val_vids)
    train_df = filtered[~filtered['video_id'].isin(val_vid_set)].copy()
    val_df = filtered[filtered['video_id'].isin(val_vid_set)].copy()

    print(f"   Custom split: {len(train_df)} train / {len(val_df)} val segments")
    print(f"   Train videos: {len(train_vids)}, Val videos: {len(val_vids)}")
    print(f"   Val videos: {sorted(val_vids)}")

    return train_df, val_df


# ============================================================
# Step 2: Build tar index for fast frame access
# ============================================================
def build_tar_index(tar_path):
    """Build a dict mapping frame filenames to their tar member info.

    Returns: {frame_filename: TarInfo}
    """
    index = {}
    with tarfile.open(tar_path, 'r') as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith('.jpg'):
                # member.name is like "P01_101/frame_0000000001.jpg"
                basename = os.path.basename(member.name)
                index[basename] = member
    return index


def build_all_tar_indices(dataset_dir, participants):
    """Build frame indices for all tarballs, grouped by video_id.

    Returns: {video_id: (tar_path, {frame_name: TarInfo})}
    """
    print("\n🔍 Indexing tar archives (one-time scan)...")
    video_index = {}
    total_tars = 0

    for pid in participants:
        tar_dir = os.path.join(dataset_dir, pid, 'rgb_frames')
        if not os.path.isdir(tar_dir):
            print(f"   ⚠️  {pid}: no rgb_frames directory found!")
            continue

        tar_files = sorted(glob.glob(os.path.join(tar_dir, '*.tar')))
        print(f"   {pid}: {len(tar_files)} tar files")
        total_tars += len(tar_files)

        for tar_path in tqdm(tar_files, desc=f"   Indexing {pid}", leave=False):
            # Video ID from tar filename: P01_101.tar -> P01_101
            video_id = os.path.splitext(os.path.basename(tar_path))[0]
            frame_index = build_tar_index(tar_path)
            video_index[video_id] = (tar_path, frame_index)

    print(f"   ✅ Indexed {total_tars} tar files, "
          f"{len(video_index)} videos, "
          f"{sum(len(v[1]) for v in video_index.values())} frames total")
    return video_index


# ============================================================
# Step 3: Extract clips
# ============================================================
def read_frames_from_tar(tar_path, frame_index, frame_names):
    """Read specific frames from a tar archive using pre-built index.

    Args:
        tar_path: path to .tar file
        frame_index: {frame_name: TarInfo} from build_tar_index
        frame_names: list of frame filenames to read

    Returns:
        list of numpy arrays (H, W, C) uint8 RGB, or None if any missing
    """
    frames = []
    with tarfile.open(tar_path, 'r') as tf:
        for fname in frame_names:
            if fname not in frame_index:
                return None
            member = frame_index[fname]
            f = tf.extractfile(member)
            if f is None:
                return None
            data = f.read()
            img = cv2.imdecode(
                np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
            frames.append(img)
    return frames


def extract_clips(annotations_df, split_name, video_index, output_dir):
    """Extract 16-frame clips from tar archives for one split."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    done_marker = os.path.join(split_dir, '.processing_complete')
    if os.path.exists(done_marker):
        existing = len(glob.glob(os.path.join(split_dir, '*.npz')))
        print(f"   ✅ {split_name}: {existing} clips already processed.")
        return existing

    # Filter to our participants
    available_pids = set(PARTICIPANTS)
    df = annotations_df[annotations_df['participant_id'].isin(available_pids)].copy()
    print(f"   Processing {len(df)} {split_name} segments "
          f"from {df['participant_id'].nunique()} participants...")

    processed = 0
    skipped = 0
    skipped_reasons = {'no_video': 0, 'missing_frames': 0, 'read_error': 0}

    # Group by video_id for efficient tar access (open each tar once)
    grouped = df.groupby('video_id')

    for video_id, group in tqdm(grouped, desc=f'{split_name}',
                                 total=len(grouped)):
        if video_id not in video_index:
            skipped += len(group)
            skipped_reasons['no_video'] += len(group)
            continue

        tar_path, frame_index = video_index[video_id]

        # Open tar once for all segments in this video
        try:
            with tarfile.open(tar_path, 'r') as tf:
                for _, row in group.iterrows():
                    narration_id = row['narration_id']
                    out_path = os.path.join(split_dir, f'{narration_id}.npz')

                    if os.path.exists(out_path):
                        processed += 1
                        continue

                    start_frame = row['start_frame']
                    stop_frame = row['stop_frame']
                    verb_class = row['verb_class']

                    # Uniformly sample NUM_FRAMES from [start, stop]
                    total = stop_frame - start_frame + 1
                    if total < NUM_FRAMES:
                        indices = np.linspace(start_frame, stop_frame,
                                              NUM_FRAMES, dtype=int)
                    else:
                        indices = np.linspace(start_frame, stop_frame,
                                              NUM_FRAMES, endpoint=False,
                                              dtype=int)

                    # Build frame filenames
                    frame_names = [f'frame_{idx:010d}.jpg' for idx in indices]

                    # Check all frames exist in index
                    if not all(fn in frame_index for fn in frame_names):
                        skipped += 1
                        skipped_reasons['missing_frames'] += 1
                        continue

                    # Read frames from the already-open tar
                    frames = []
                    valid = True
                    for fname in frame_names:
                        member = frame_index[fname]
                        f = tf.extractfile(member)
                        if f is None:
                            valid = False
                            break
                        data = f.read()
                        img = cv2.imdecode(
                            np.frombuffer(data, np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        if img is None:
                            valid = False
                            break
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                                         interpolation=cv2.INTER_LINEAR)
                        frames.append(img)

                    if not valid or len(frames) != NUM_FRAMES:
                        skipped += 1
                        skipped_reasons['read_error'] += 1
                        continue

                    frames_arr = np.stack(frames, axis=0)  # (T, H, W, C)
                    np.savez_compressed(out_path, frames=frames_arr,
                                        verb_class=verb_class)
                    processed += 1

        except Exception as e:
            print(f"\n   ⚠️  Error processing {video_id}: {e}")
            skipped += len(group)
            continue

    Path(done_marker).touch()
    print(f"   ✅ {split_name}: {processed} clips saved, {skipped} skipped")
    print(f"      Skip reasons: {skipped_reasons}")
    return processed


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  EPIC-KITCHENS-100 Local Dataset Processor")
    print("=" * 60)
    print(f"  Dataset dir:    {DATASET_DIR}")
    print(f"  Output dir:     {OUTPUT_ROOT}")
    print(f"  Participants:   {PARTICIPANTS}")
    print(f"  Frames/clip:    {NUM_FRAMES}")
    print(f"  Image size:     {IMG_SIZE}×{IMG_SIZE}")
    print("=" * 60)

    # Verify dataset exists
    for pid in PARTICIPANTS:
        pid_dir = os.path.join(DATASET_DIR, pid)
        if os.path.isdir(pid_dir):
            print(f"  ✅ {pid} found")
        else:
            print(f"  ❌ {pid} NOT found at {pid_dir}")
            sys.exit(1)

    # Create output directories (matches Colab Drive structure)
    for subdir in ['annotations', 'processed_clips', 'checkpoints', 'logs']:
        os.makedirs(os.path.join(OUTPUT_ROOT, subdir), exist_ok=True)

    t0 = time.time()

    # Step 1: Index tarballs (needed before annotation filtering)
    print("\n" + "─" * 60)
    print("Step 1/3: Indexing tar archives")
    print("─" * 60)
    video_index = build_all_tar_indices(DATASET_DIR, PARTICIPANTS)

    # Step 2: Annotations + custom train/val split
    print("\n" + "─" * 60)
    print("Step 2/3: Annotations & custom split")
    print("─" * 60)
    train_df, val_df = setup_annotations(video_index.keys())

    # Step 3: Extract clips
    # Remove stale processing markers so clips are re-processed
    print("\n" + "─" * 60)
    print("Step 3/3: Extracting clips")
    print("─" * 60)
    processed_dir = os.path.join(OUTPUT_ROOT, 'processed_clips')

    # Clear old markers and clips if re-running with new split
    for split in ['train', 'val']:
        marker = os.path.join(processed_dir, split, '.processing_complete')
        if os.path.exists(marker):
            os.remove(marker)
            print(f"   🔄 Cleared stale {split} marker for re-processing.")

    n_train = extract_clips(train_df, 'train', video_index, processed_dir)
    n_val = extract_clips(val_df, 'val', video_index, processed_dir)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  ✅ DONE in {elapsed/60:.1f} minutes!")
    print(f"     Train clips: {n_train}")
    print(f"     Val clips:   {n_val}")
    print(f"\n  📁 Upload '{OUTPUT_ROOT}' to Google Drive at:")
    print(f"     My Drive/cv_project/")
    print("=" * 60)


if __name__ == '__main__':
    main()
