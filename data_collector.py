"""
╔══════════════════════════════════════════════════════════════════╗
║           STAGE 1 — DATA COLLECTOR & FEATURE EXTRACTOR          ║
║  Captures MediaPipe Pose landmarks from webcam or video file     ║
║  and saves them as a labelled CSV for ML training.               ║
╚══════════════════════════════════════════════════════════════════╝

USAGE
─────
  # Collect samples for 'squat' from webcam (press Q to stop):
  python data_collector.py --label squat --source 0 --output data/landmarks.csv

  # Collect samples for 'pushup' from a pre-recorded video:
  python data_collector.py --label pushup --source videos/pushup.mp4 --output data/landmarks.csv

  # Collect with a custom FPS skip (process every 3rd frame):
  python data_collector.py --label bicep_curl --source 0 --output data/landmarks.csv --skip 3

SUPPORTED LABELS (examples):  squat | pushup | bicep_curl
──────────────────────────────────────────────────────────
Run this script once per exercise class.  Each run APPENDS rows to
the same CSV so you can build the dataset incrementally.
"""

import argparse
import csv
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────

# MediaPipe landmark indices that are most informative for full-body
# exercise classification.  We intentionally exclude face landmarks
# (0-10) to keep the feature vector compact and pose-focused.
RELEVANT_LANDMARKS = [
    11, 12,        # shoulders
    13, 14,        # elbows
    15, 16,        # wrists
    23, 24,        # hips
    25, 26,        # knees
    27, 28,        # ankles
    29, 30,        # heels
    31, 32,        # foot index
]

# Each landmark contributes x, y, z → 3 features
NUM_FEATURES = len(RELEVANT_LANDMARKS) * 3

# Column header for the CSV
FEATURE_COLUMNS = []
for idx in RELEVANT_LANDMARKS:
    FEATURE_COLUMNS += [f"lm{idx}_x", f"lm{idx}_y", f"lm{idx}_z"]
FEATURE_COLUMNS.append("label")

# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────

def extract_features(landmarks) -> list[float] | None:
    """
    Given a MediaPipe NormalizedLandmarkList, return a flat list of
    [x, y, z] values for RELEVANT_LANDMARKS only.

    Returns None if landmarks are missing / confidence is too low.
    """
    if landmarks is None:
        return None

    lm_list = landmarks.landmark
    row = []
    for idx in RELEVANT_LANDMARKS:
        lm = lm_list[idx]
        row.extend([lm.x, lm.y, lm.z])
    return row


def draw_overlay(frame: np.ndarray, label: str, sample_count: int,
                 fps: float, collecting: bool) -> np.ndarray:
    """Render a HUD on the frame so the operator knows what's happening."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Status banner
    colour = (0, 200, 80) if collecting else (0, 80, 220)
    cv2.rectangle(overlay, (0, 0), (w, 55), colour, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    status = "COLLECTING" if collecting else "PAUSED  (press SPACE)"
    cv2.putText(frame, f"Exercise: {label.upper()}   [{status}]",
                (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    # Sample counter & FPS
    cv2.putText(frame, f"Samples: {sample_count}   FPS: {fps:.1f}",
                (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 255), 1, cv2.LINE_AA)

    # Key guide
    guide = "SPACE=pause/resume   Q=quit & save"
    cv2.putText(frame, guide, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def ensure_csv(path: str) -> None:
    """Create the CSV with headers if it does not already exist."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_COLUMNS)
        print(f"[INFO] Created new dataset file: {path}")
    else:
        print(f"[INFO] Appending to existing dataset: {path}")


# ──────────────────────────────────────────────────────────────────
# MAIN COLLECTION LOOP
# ──────────────────────────────────────────────────────────────────

def run_collector(label: str, source, output: str, skip: int) -> None:
    ensure_csv(output)

    # ── MediaPipe setup ──────────────────────────────────────────
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # ── Video capture ────────────────────────────────────────────
    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    print(f"\n[INFO] Starting collection for label='{label}'")
    print("[INFO]  Press SPACE to toggle pause/resume")
    print("[INFO]  Press  Q   to quit and save\n")

    sample_count = 0
    frame_idx    = 0
    collecting   = True
    prev_time    = time.time()

    with open(output, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video source reached.")
                break

            frame_idx += 1

            # FPS calculation
            cur_time = time.time()
            fps = 1.0 / max(cur_time - prev_time, 1e-6)
            prev_time = cur_time

            # ── Pose detection ───────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # Draw skeleton overlay
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
                )

            # ── Feature extraction & saving ──────────────────────
            if collecting and (frame_idx % skip == 0) and results.pose_landmarks:
                features = extract_features(results.pose_landmarks)
                if features:
                    writer.writerow(features + [label])
                    sample_count += 1

            # ── HUD overlay ──────────────────────────────────────
            frame = draw_overlay(frame, label, sample_count, fps, collecting)
            cv2.imshow("Data Collector — AI Fitness Tracker", frame)

            # ── Key handling ─────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                collecting = not collecting
                status = "RESUMED" if collecting else "PAUSED"
                print(f"[INFO] Collection {status}  (total so far: {sample_count})")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    print(f"\n[DONE] Saved {sample_count} samples for label='{label}' → {output}")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect MediaPipe pose-landmark samples for exercise classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--label",  required=True,
                   help="Exercise label to record  (e.g. squat, pushup, bicep_curl)")
    p.add_argument("--source", default="0",
                   help="Webcam index (0, 1…) or path to a video file  [default: 0]")
    p.add_argument("--output", default="data/landmarks.csv",
                   help="Path to the output CSV file  [default: data/landmarks.csv]")
    p.add_argument("--skip",   type=int, default=2,
                   help="Process every N-th frame (reduces redundant samples)  [default: 2]")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_collector(
        label=args.label.lower().strip(),
        source=args.source,
        output=args.output,
        skip=args.skip,
    )
