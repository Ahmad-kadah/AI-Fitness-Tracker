"""
╔══════════════════════════════════════════════════════════════════╗
║       STAGE 3 — REAL-TIME INFERENCE, REP COUNTING & FORM HUD    ║
║  Loads the trained model, classifies exercise frame-by-frame,    ║
║  smooths predictions, then dispatches to per-exercise logic for  ║
║  rep counting and form-quality feedback.                         ║
╚══════════════════════════════════════════════════════════════════╝

USAGE
─────
  # Run on webcam (index 0):
  python main_tracker.py --source 0 --models models/

  # Run on a pre-recorded video and save annotated output:
  python main_tracker.py --source videos/workout.mp4 --models models/ --save output.mp4

  # Force a specific exercise (skip ML classification):
  python main_tracker.py --source 0 --models models/ --force-exercise squat

  # Adjust prediction smoothing window (default 10):
  python main_tracker.py --source 0 --models models/ --smooth 15
"""

import argparse
import collections
import math
import os
import pickle
import sys
import time
from pathlib import Path
from statistics import mode as stat_mode
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ──────────────────────────────────────────────────────────────────
# LANDMARK INDEX MAP  (MediaPipe BlazePose 33-point model)
# ──────────────────────────────────────────────────────────────────
LM = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow":    13, "r_elbow":    14,
    "l_wrist":    15, "r_wrist":    16,
    "l_hip":      23, "r_hip":      24,
    "l_knee":     25, "r_knee":     26,
    "l_ankle":    27, "r_ankle":    28,
}

# Features must match data_collector.py
RELEVANT_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


# ──────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────

def _lm_coords(landmarks, idx: int) -> np.ndarray:
    """Return [x, y] pixel-normalised [0-1] for a landmark index."""
    lm = landmarks.landmark[idx]
    return np.array([lm.x, lm.y])


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the interior angle at vertex B formed by points A-B-C.
    Returns degrees in [0, 180].
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


# ──────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (must match data_collector.py exactly)
# ──────────────────────────────────────────────────────────────────

def extract_features(landmarks) -> Optional[np.ndarray]:
    if landmarks is None:
        return None
    lm_list = landmarks.landmark
    row = []
    for idx in RELEVANT_LANDMARKS:
        lm = lm_list[idx]
        row.extend([lm.x, lm.y, lm.z])
    return np.array(row, dtype=np.float32).reshape(1, -1)


# ──────────────────────────────────────────────────────────────────
# BASE EXERCISE CLASS
# ──────────────────────────────────────────────────────────────────

class ExerciseAnalyser:
    """
    Abstract base for each exercise.  Subclass and implement:
      - `analyse(landmarks) → dict`   (must return a dict with keys below)

    Expected dict keys returned by analyse():
      - "rep_count"   : int
      - "stage"       : str   e.g. "up" / "down"
      - "feedback"    : list[str]   form cues (empty = perfect)
      - "angles"      : dict[str, float]   drawn on screen
    """
    name: str = "unknown"

    def reset(self):
        self.rep_count = 0
        self.stage     = None

    def analyse(self, lm) -> dict:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────
# SQUAT ANALYSER
# ──────────────────────────────────────────────────────────────────

class SquatAnalyser(ExerciseAnalyser):
    """
    Rep counting  : hip descends until knee angle < DOWN_THRESH,
                    then rises back until knee angle > UP_THRESH.
    Form checks   : knee cave (knee tracks over foot), back rounding
                    (hip-shoulder vertical angle), depth check.
    """
    name        = "squat"
    DOWN_THRESH = 100   # degrees – knee bent ≥ this → "down"
    UP_THRESH   = 160   # degrees – knee extended ≥ this → "up"

    def __init__(self):
        self.rep_count = 0
        self.stage     = None

    def analyse(self, lm) -> dict:
        feedback = []
        angles   = {}

        # ── Key points ───────────────────────────────────────────
        l_hip    = _lm_coords(lm, LM["l_hip"])
        r_hip    = _lm_coords(lm, LM["r_hip"])
        l_knee   = _lm_coords(lm, LM["l_knee"])
        r_knee   = _lm_coords(lm, LM["r_knee"])
        l_ankle  = _lm_coords(lm, LM["l_ankle"])
        r_ankle  = _lm_coords(lm, LM["r_ankle"])
        l_shoul  = _lm_coords(lm, LM["l_shoulder"])
        r_shoul  = _lm_coords(lm, LM["r_shoulder"])

        # ── Knee angles ──────────────────────────────────────────
        l_knee_angle = angle_between(l_hip, l_knee, l_ankle)
        r_knee_angle = angle_between(r_hip, r_knee, r_ankle)
        avg_knee     = (l_knee_angle + r_knee_angle) / 2.0
        angles["L-Knee"] = l_knee_angle
        angles["R-Knee"] = r_knee_angle

        # ── Rep counting state machine ───────────────────────────
        if avg_knee < self.DOWN_THRESH:
            self.stage = "down"
        if avg_knee > self.UP_THRESH and self.stage == "down":
            self.stage = "up"
            self.rep_count += 1

        # ── Back angle (torso lean) ───────────────────────────────
        mid_hip    = midpoint(l_hip, r_hip)
        mid_shoul  = midpoint(l_shoul, r_shoul)
        # Vertical reference point directly above mid_hip
        vert_ref   = mid_hip + np.array([0, -0.5])
        back_angle = angle_between(vert_ref, mid_hip, mid_shoul)
        angles["Back"] = back_angle

        if back_angle > 55:
            feedback.append("⚠ Keep your back straighter")

        # ── Knee cave check (x-coordinate of knee vs ankle) ──────
        if l_knee[0] < l_ankle[0] - 0.04:
            feedback.append("⚠ Left knee caving in — push knees out")
        if r_knee[0] > r_ankle[0] + 0.04:
            feedback.append("⚠ Right knee caving in — push knees out")

        # ── Depth check ──────────────────────────────────────────
        if self.stage == "down" and avg_knee > 115:
            feedback.append("⚠ Go deeper — break parallel")

        return {
            "rep_count": self.rep_count,
            "stage":     self.stage or "-",
            "feedback":  feedback,
            "angles":    angles,
        }


# ──────────────────────────────────────────────────────────────────
# PUSH-UP ANALYSER
# ──────────────────────────────────────────────────────────────────

class PushupAnalyser(ExerciseAnalyser):
    """
    Rep counting  : elbow angle < DOWN_THRESH → "down", then > UP_THRESH → "up" (+1 rep).
    Form checks   : body alignment (hip sag / pike), elbow flare.
    """
    name        = "pushup"
    DOWN_THRESH = 90
    UP_THRESH   = 155

    def __init__(self):
        self.rep_count = 0
        self.stage     = None

    def analyse(self, lm) -> dict:
        feedback = []
        angles   = {}

        l_shoul = _lm_coords(lm, LM["l_shoulder"])
        r_shoul = _lm_coords(lm, LM["r_shoulder"])
        l_elbow = _lm_coords(lm, LM["l_elbow"])
        r_elbow = _lm_coords(lm, LM["r_elbow"])
        l_wrist = _lm_coords(lm, LM["l_wrist"])
        r_wrist = _lm_coords(lm, LM["r_wrist"])
        l_hip   = _lm_coords(lm, LM["l_hip"])
        r_hip   = _lm_coords(lm, LM["r_hip"])
        l_knee  = _lm_coords(lm, LM["l_knee"])
        r_knee  = _lm_coords(lm, LM["r_knee"])

        # ── Elbow angles ─────────────────────────────────────────
        l_elbow_angle = angle_between(l_shoul, l_elbow, l_wrist)
        r_elbow_angle = angle_between(r_shoul, r_elbow, r_wrist)
        avg_elbow     = (l_elbow_angle + r_elbow_angle) / 2.0
        angles["L-Elbow"] = l_elbow_angle
        angles["R-Elbow"] = r_elbow_angle

        # ── Rep counting ─────────────────────────────────────────
        if avg_elbow < self.DOWN_THRESH:
            self.stage = "down"
        if avg_elbow > self.UP_THRESH and self.stage == "down":
            self.stage = "up"
            self.rep_count += 1

        # ── Body alignment (shoulder–hip–knee should be ~180°) ───
        mid_shoul = midpoint(l_shoul, r_shoul)
        mid_hip   = midpoint(l_hip, r_hip)
        mid_knee  = midpoint(l_knee, r_knee)
        body_line = angle_between(mid_shoul, mid_hip, mid_knee)
        angles["Body"] = body_line

        if body_line < 160:
            if mid_hip[1] < mid_shoul[1]:
                feedback.append("⚠ Hips too high — lower them")
            else:
                feedback.append("⚠ Hips sagging — engage your core")

        # ── Elbow flare (elbow should not exceed ~75° from torso) ─
        elbow_flare_l = angle_between(r_shoul, l_shoul, l_elbow)
        if elbow_flare_l > 80:
            feedback.append("⚠ Elbows flaring — tuck them closer")

        return {
            "rep_count": self.rep_count,
            "stage":     self.stage or "-",
            "feedback":  feedback,
            "angles":    angles,
        }


# ──────────────────────────────────────────────────────────────────
# BICEP CURL ANALYSER
# ──────────────────────────────────────────────────────────────────

class BicepCurlAnalyser(ExerciseAnalyser):
    """
    Rep counting  : average elbow angle < DOWN_THRESH → "down",
                    > UP_THRESH → "up" (+1 rep).
    Form checks   : shoulder stability (upper arm should stay vertical),
                    full extension at bottom, full contraction at top.
    """
    name        = "bicep_curl"
    DOWN_THRESH = 45    # curl fully contracted
    UP_THRESH   = 155   # arm fully extended

    def __init__(self):
        self.rep_count = 0
        self.stage     = None

    def analyse(self, lm) -> dict:
        feedback = []
        angles   = {}

        l_shoul = _lm_coords(lm, LM["l_shoulder"])
        r_shoul = _lm_coords(lm, LM["r_shoulder"])
        l_elbow = _lm_coords(lm, LM["l_elbow"])
        r_elbow = _lm_coords(lm, LM["r_elbow"])
        l_wrist = _lm_coords(lm, LM["l_wrist"])
        r_wrist = _lm_coords(lm, LM["r_wrist"])
        l_hip   = _lm_coords(lm, LM["l_hip"])
        r_hip   = _lm_coords(lm, LM["r_hip"])

        # ── Elbow angles ─────────────────────────────────────────
        l_elbow_angle = angle_between(l_shoul, l_elbow, l_wrist)
        r_elbow_angle = angle_between(r_shoul, r_elbow, r_wrist)
        avg_elbow     = (l_elbow_angle + r_elbow_angle) / 2.0
        angles["L-Elbow"] = l_elbow_angle
        angles["R-Elbow"] = r_elbow_angle

        # ── Rep counting ─────────────────────────────────────────
        if avg_elbow < self.DOWN_THRESH:
            self.stage = "up"            # fully curled = "up" stage
        if avg_elbow > self.UP_THRESH and self.stage == "up":
            self.stage = "down"          # fully extended = "down" → counted
            self.rep_count += 1

        # ── Shoulder stability — upper arm should be ~vertical ───
        # Upper arm vector
        l_upper = l_elbow - l_shoul
        r_upper = r_elbow - r_shoul
        vert    = np.array([0, 1])      # pointing down in image coords
        l_swing = math.degrees(math.acos(
            np.clip(np.dot(l_upper / (np.linalg.norm(l_upper) + 1e-9), vert), -1, 1)
        ))
        r_swing = math.degrees(math.acos(
            np.clip(np.dot(r_upper / (np.linalg.norm(r_upper) + 1e-9), vert), -1, 1)
        ))
        if l_swing > 30 or r_swing > 30:
            feedback.append("⚠ Keep upper arms still — don't swing")

        # ── Full range check ─────────────────────────────────────
        if self.stage == "up" and avg_elbow > 60:
            feedback.append("⚠ Curl higher — full contraction")
        if self.stage == "down" and avg_elbow < 145:
            feedback.append("⚠ Extend fully at the bottom")

        return {
            "rep_count": self.rep_count,
            "stage":     self.stage or "-",
            "feedback":  feedback,
            "angles":    angles,
        }


# ──────────────────────────────────────────────────────────────────
# EXERCISE REGISTRY  — add new exercises here
class PullupAnalyser(ExerciseAnalyser):
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                      PULL-UP ANALYSER                           ║
    ║                                                                  ║
    ║  REP CYCLE                                                       ║
    ║  ─────────                                                       ║
    ║  "down"  →  Dead hang / full arm extension:                     ║
    ║             elbow angle ≥ DOWN_THRESH  (~160°)                  ║
    ║             (shoulder fully depressed, scapula elevated)         ║
    ║  "up"    →  Chin clears bar / peak contraction:                 ║
    ║             elbow angle ≤ UP_THRESH    (~65°)                   ║
    ║             (chin above wrist level, elbows drive to hips)       ║
    ║                                                                  ║
    ║  ⚠  INVERSION NOTE                                              ║
    ║  In a pull-up the "up" position has a SMALLER elbow angle       ║
    ║  (arms bent) and the "down" position has a LARGER elbow angle   ║
    ║  (arms straight).  Thresholds are intentionally reversed vs     ║
    ║  pushing exercises.                                              ║
    ║                                                                  ║
    ║  JOINTS TRACKED                                                  ║
    ║  ──────────────                                                  ║
    ║  • Elbow angle       → shoulder–elbow–wrist  (primary driver)   ║
    ║  • Shoulder angle    → hip–shoulder–elbow    (scapular path)    ║
    ║  • Torso lean        → vertical–mid_hip–mid_shoulder            ║
    ║  • Hip angle         → shoulder–hip–knee     (kipping / swing)  ║
    ║  • Elbow symmetry    → |L_elbow − R_elbow|   (imbalance)        ║
    ║                                                                  ║
    ║  FORM CHECKS  (5 biomechanical cues)                            ║
    ║  ────────────────────────────────────                            ║
    ║  1. Chin not clearing bar   (wrist y vs shoulder y at peak)     ║
    ║  2. Kipping / hip swing     (hip angle deviation mid-rep)       ║
    ║  3. Excessive torso lean    (body swinging behind vertical)      ║
    ║  4. Incomplete dead hang    (elbow not fully extending at        ║
    ║                              bottom — short range of motion)    ║
    ║  5. Unilateral elbow        (L vs R elbow angle > 15°)          ║
    ║     asymmetry               (imbalanced lat/bicep engagement)   ║
    ╚══════════════════════════════════════════════════════════════════╝

    CAMERA SETUP NOTE
    ─────────────────
    Best captured from a FRONT or SLIGHT SIDE view so that both
    elbows, wrists, and shoulders are clearly visible against a
    clean background.  The bar should be above the top edge of
    the frame or at the very top so the full hang is captured.
    A side-angle camera also reveals torso lean and kipping.
    """

    name = "pullup"

    # ── Rep-counting thresholds (elbow angle in degrees) ───────────
    # NOTE: Logic is INVERTED compared to pushing exercises.
    #       Arms BEND going "up" → smaller angle = contracted position.
    #       Arms EXTEND going "down" → larger angle = dead hang.
    DOWN_THRESH = 160   # elbow ≥ this  →  dead hang confirmed  ("down")
    UP_THRESH   = 65    # elbow ≤ this  →  chin over bar / peak ("up")

    # ── Form-check tolerances ───────────────────────────────────────
    CHIN_CLEAR_MARGIN     = 0.03   # shoulder.y must be ≥ wrist.y − margin
                                   # (in normalised coords; smaller y = higher)
    KIPPING_HIP_WARN      = 25     # degrees deviation of hip angle from neutral
    TORSO_LEAN_WARN       = 20     # degrees torso from vertical
    DEAD_HANG_WARN        = 150    # elbow must reach at least this at bottom
    SYMMETRY_WARN         = 15     # degrees L vs R elbow difference

    # Neutral hip angle captured on first "down" frame for kipping baseline
    _neutral_hip_angle    = None

    def __init__(self):
        self.rep_count         = 0
        self.stage             = None
        self._neutral_hip      = None   # established at first dead-hang frame
        self._hang_elbow_max   = 0.0    # tracks maximum elbow extension each rep

    def reset(self):
        self.rep_count       = 0
        self.stage           = None
        self._neutral_hip    = None
        self._hang_elbow_max = 0.0

    # ─────────────────────────────────────────────────────────────────
    def analyse(self, lm) -> dict:
        """
        Parameters
        ----------
        lm : mediapipe NormalizedLandmarkList

        Returns
        -------
        dict with keys: rep_count, stage, feedback, angles
        """
        feedback = []
        angles   = {}

        # ── 1. Gather key coordinates ──────────────────────────────
        l_shoulder = _lm_coords(lm, LM["l_shoulder"])
        r_shoulder = _lm_coords(lm, LM["r_shoulder"])
        l_elbow    = _lm_coords(lm, LM["l_elbow"])
        r_elbow    = _lm_coords(lm, LM["r_elbow"])
        l_wrist    = _lm_coords(lm, LM["l_wrist"])
        r_wrist    = _lm_coords(lm, LM["r_wrist"])
        l_hip      = _lm_coords(lm, LM["l_hip"])
        r_hip      = _lm_coords(lm, LM["r_hip"])
        l_knee     = _lm_coords(lm, LM["l_knee"])
        r_knee     = _lm_coords(lm, LM["r_knee"])

        mid_shoulder = midpoint(l_shoulder, r_shoulder)
        mid_hip      = midpoint(l_hip,      r_hip)
        mid_wrist    = midpoint(l_wrist,    r_wrist)
        mid_knee     = midpoint(l_knee,     r_knee)

        # ── 2. Joint angle calculations ────────────────────────────

        # PRIMARY: Elbow flexion — shoulder–elbow–wrist
        # Dead hang  → ~170-180°  (fully extended biceps/brachialis)
        # Peak pull  → ~50-70°    (full lat/bicep contraction)
        l_elbow_angle = angle_between(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = angle_between(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        angles["L-Elbow"] = l_elbow_angle
        angles["R-Elbow"] = r_elbow_angle

        # SECONDARY: Shoulder angle — hip–shoulder–elbow
        # Tracks how far the elbow travels behind the torso at peak
        # contraction; elbows should drive DOWN and BACK (not out).
        # Also reveals scapular depression — critical for lat engagement.
        l_shoulder_angle = angle_between(l_hip, l_shoulder, l_elbow)
        r_shoulder_angle = angle_between(r_hip, r_shoulder, r_elbow)
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0
        angles["Shoulder"] = avg_shoulder_angle

        # TERTIARY: Torso lean from vertical
        # Reference point directly above mid_hip in image space.
        # (y decreases upward; subtract y to get a "higher" point)
        vertical_ref = mid_hip + np.array([0.0, -0.5])
        torso_angle  = angle_between(vertical_ref, mid_hip, mid_shoulder)
        angles["Torso"] = torso_angle

        # QUATERNARY: Hip angle — shoulder–hip–knee
        # In a strict pull-up this stays nearly constant (~170-180°,
        # body hangs straight).  A large deviation mid-rep = kipping.
        l_hip_angle  = angle_between(l_shoulder, l_hip, l_knee)
        r_hip_angle  = angle_between(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2.0
        angles["Hip"] = avg_hip_angle

        # ── 3. Rep-counting state machine ──────────────────────────
        #
        # INVERTED logic vs pressing exercises:
        #   avg_elbow ≥ DOWN_THRESH  →  dead hang  ("down")
        #   avg_elbow ≤ UP_THRESH    →  chin up     ("up")
        #
        # Track maximum elbow extension reached during the descent
        # so CHECK 4 (incomplete dead hang) can fire accurately.
        #
        if avg_elbow_angle >= self.DOWN_THRESH:
            self.stage           = "down"
            self._hang_elbow_max = max(self._hang_elbow_max, avg_elbow_angle)
            # Capture neutral hip angle on the very first confirmed hang
            if self._neutral_hip is None:
                self._neutral_hip = avg_hip_angle

        if avg_elbow_angle <= self.UP_THRESH and self.stage == "down":
            self.stage           = "up"
            self.rep_count      += 1
            self._hang_elbow_max = 0.0   # reset for next descent

        # ── 4. Form checks ─────────────────────────────────────────

        # ── CHECK 1: Chin not clearing the bar ─────────────────────
        # At peak contraction (stage == "up") the chin / eyes should
        # clear the bar, proxied by: mid_shoulder.y ≤ mid_wrist.y
        # (in normalised image coords smaller y = higher in frame).
        # If the shoulder is still BELOW the wrists the lifter has
        # not pulled high enough — common fault on fatigue sets.
        if self.stage == "up":
            # mid_shoulder.y < mid_wrist.y means shoulders are ABOVE wrists
            chin_clearance = mid_wrist[1] - mid_shoulder[1]
            if chin_clearance < self.CHIN_CLEAR_MARGIN:
                feedback.append(
                    "⚠ Chin not clearing bar — pull higher until chin is above hands"
                )

        # ── CHECK 2: Kipping / hip swing ───────────────────────────
        # A strict pull-up keeps the body plumb throughout.
        # We compare current hip angle against the neutral baseline
        # captured at the first dead hang; > KIPPING_HIP_WARN degrees
        # of deviation mid-rep signals momentum-driven kipping.
        if self._neutral_hip is not None and self.stage != "down":
            hip_deviation = abs(avg_hip_angle - self._neutral_hip)
            if hip_deviation > self.KIPPING_HIP_WARN:
                feedback.append(
                    "⚠ Kipping detected — keep body straight, no hip swing"
                )

        # ── CHECK 3: Excessive torso lean / body swing ─────────────
        # The torso should remain roughly vertical (±~15°) throughout.
        # Leaning far back converts the pull-up into a row-like motion,
        # reducing lat activation and stressing the lumbar spine.
        if torso_angle > self.TORSO_LEAN_WARN:
            feedback.append(
                "⚠ Torso leaning back too far — keep body vertical"
            )

        # ── CHECK 4: Incomplete dead hang at the bottom ─────────────
        # Stopping the descent before full elbow extension cheats the
        # full range of motion, reduces long-head bicep and lat stretch,
        # and inflates rep counts with partial reps.
        # We evaluate this only when transitioning INTO the "down" stage
        # (i.e., the elbow just reached DOWN_THRESH after a rep) by
        # checking whether the tracked max elbow angle is sufficient.
        if self.stage == "down" and 0 < self._hang_elbow_max < self.DEAD_HANG_WARN:
            feedback.append(
                "⚠ Not fully hanging — extend arms completely at the bottom"
            )

        # ── CHECK 5: Left-right elbow asymmetry ────────────────────
        # A persistent angle gap between the two elbows means one lat /
        # bicep is carrying more load — a precursor to shoulder or elbow
        # overuse injuries and visible postural asymmetry over time.
        elbow_diff = abs(l_elbow_angle - r_elbow_angle)
        if elbow_diff > self.SYMMETRY_WARN:
            lagging = "right" if l_elbow_angle > r_elbow_angle else "left"
            feedback.append(
                f"⚠ Uneven pull — {lagging} arm lagging, engage both lats equally"
            )

        # ── 5. Return standardised result dict ─────────────────────
        return {
            "rep_count": self.rep_count,
            "stage":     self.stage or "-",
            "feedback":  feedback,
            "angles":    angles,
        }

class InclineChestPressAnalyser(ExerciseAnalyser):
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║               INCLINE CHEST PRESS ANALYSER                      ║
    ║                                                                  ║
    ║  REP CYCLE                                                       ║
    ║  ─────────                                                       ║
    ║  "down"  →  Bar lowered to upper chest:                         ║
    ║             elbow angle ≤ DOWN_THRESH  (~70-80°)                ║
    ║             (elbows below shoulder plane, full stretch)          ║
    ║  "up"    →  Arms fully pressed / locked out:                    ║
    ║             elbow angle ≥ UP_THRESH    (~155-165°)              ║
    ║                                                                  ║
    ║  JOINTS TRACKED                                                  ║
    ║  ──────────────                                                  ║
    ║  • Elbow angle      → shoulder–elbow–wrist  (primary driver)    ║
    ║  • Shoulder angle   → hip–shoulder–elbow    (elbow flare/path)  ║
    ║  • Wrist alignment  → wrist x ≈ elbow x     (bar path proxy)   ║
    ║  • Elbow symmetry   → L elbow angle vs R     (imbalance check)  ║
    ║                                                                  ║
    ║  FORM CHECKS  (5 biomechanical cues)                            ║
    ║  ────────────────────────────────────                            ║
    ║  1. Excessive elbow flare  (shoulder angle > 90°)               ║
    ║  2. Wrist misalignment     (wrist drifts behind/over elbow)     ║
    ║  3. Unilateral imbalance   (L vs R elbow angle > 15°)           ║
    ║  4. Incomplete lockout     (elbow < UP_THRESH at "up" stage)    ║
    ║  5. Incomplete descent     (elbow > DOWN_THRESH + 15° at        ║
    ║                             "down" stage — not lowering enough) ║
    ╚══════════════════════════════════════════════════════════════════╝

    CAMERA SETUP NOTE
    ─────────────────
    Best captured from a SIDE VIEW (sagittal plane) or a slight
    front-angle so both elbows and wrists are clearly visible.
    The athlete is lying on an incline bench (~30-45°), pressing
    upward at that incline angle.
    """

    name = "incline_chest_press"

    # ── Rep-counting thresholds (elbow angle in degrees) ───────────
    DOWN_THRESH = 80    # elbow ≤ this  →  bar at chest / eccentric bottom
    UP_THRESH   = 155   # elbow ≥ this  →  arms locked out / concentric top

    # ── Form-check tolerances ───────────────────────────────────────
    ELBOW_FLARE_WARN      = 90    # shoulder–elbow angle beyond this = flaring
    WRIST_STACK_WARN      = 0.06  # normalised units; wrist x vs elbow x offset
    SYMMETRY_WARN         = 15    # degrees difference L vs R elbow = imbalance
    INCOMPLETE_PRESS_WARN = 145   # if "up" triggered but elbow below this
    INCOMPLETE_LOWER_WARN = 95    # if "down" triggered but elbow above this

    def __init__(self):
        self.rep_count = 0
        self.stage     = None

    def reset(self):
        self.rep_count = 0
        self.stage     = None

    # ─────────────────────────────────────────────────────────────────
    def analyse(self, lm) -> dict:
        """
        Parameters
        ----------
        lm : mediapipe NormalizedLandmarkList

        Returns
        -------
        dict with keys: rep_count, stage, feedback, angles
        """
        feedback = []
        angles   = {}

        # ── 1. Gather key coordinates ──────────────────────────────
        l_shoulder = _lm_coords(lm, LM["l_shoulder"])
        r_shoulder = _lm_coords(lm, LM["r_shoulder"])
        l_elbow    = _lm_coords(lm, LM["l_elbow"])
        r_elbow    = _lm_coords(lm, LM["r_elbow"])
        l_wrist    = _lm_coords(lm, LM["l_wrist"])
        r_wrist    = _lm_coords(lm, LM["r_wrist"])
        l_hip      = _lm_coords(lm, LM["l_hip"])
        r_hip      = _lm_coords(lm, LM["r_hip"])

        mid_shoulder = midpoint(l_shoulder, r_shoulder)
        mid_hip      = midpoint(l_hip,      r_hip)

        # ── 2. Joint angle calculations ────────────────────────────

        # PRIMARY: Elbow flexion/extension — shoulder–elbow–wrist
        # This is the main rep-counting angle.
        # ~70-80°  = bar lowered to upper chest (bottom of press)
        # ~155-165° = arms fully extended overhead (top lockout)
        l_elbow_angle = angle_between(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = angle_between(r_shoulder, r_elbow, r_wrist)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2.0
        angles["L-Elbow"] = l_elbow_angle
        angles["R-Elbow"] = r_elbow_angle

        # SECONDARY: Shoulder angle — hip–shoulder–elbow
        # On an incline press the upper arm should travel at ~45-75°
        # relative to the torso (tucked, not flared).
        # A large hip–shoulder–elbow angle signals elbow flare, which
        # shifts stress from the pec major to the anterior deltoid
        # and strains the shoulder capsule/rotator cuff.
        l_shoulder_angle = angle_between(l_hip, l_shoulder, l_elbow)
        r_shoulder_angle = angle_between(r_hip, r_shoulder, r_elbow)
        avg_shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2.0
        angles["L-Shoulder"] = l_shoulder_angle
        angles["R-Shoulder"] = r_shoulder_angle

        # TERTIARY: Wrist-over-elbow stacking
        # In a mechanically sound press the wrist should stay
        # roughly stacked directly above the elbow (same x position
        # when viewed from the side / slight front angle).
        # Significant forward or backward wrist drift stresses the
        # wrist joint and destabilises the bar path.
        # We compute per-side x-axis offset (wrist x − elbow x).
        l_wrist_drift = abs(l_wrist[0] - l_elbow[0])
        r_wrist_drift = abs(r_wrist[0] - r_elbow[0])

        # ── 3. Rep-counting state machine ──────────────────────────
        #
        #  "down" is confirmed when elbows are sufficiently bent
        #  (eccentric/loading phase, bar near upper chest).
        #  A rep is only counted when the lifter RETURNS to lockout
        #  ("up") after being in the "down" state — ruling out
        #  partial-range bouncing or false triggers at rest.
        #
        if avg_elbow_angle <= self.DOWN_THRESH:
            self.stage = "down"

        if avg_elbow_angle >= self.UP_THRESH and self.stage == "down":
            self.stage = "up"
            self.rep_count += 1

        # ── 4. Form checks ─────────────────────────────────────────

        # ── CHECK 1: Elbow flare ────────────────────────────────────
        # On an incline press the elbows should stay at ~45-75° from
        # the torso.  When the shoulder angle (hip–shoulder–elbow)
        # exceeds ~90° the upper arm is perpendicular to the torso
        # (like a flat "T"), which removes pec tension, impinges the
        # shoulder, and greatly increases anterior capsule stress.
        if avg_shoulder_angle > self.ELBOW_FLARE_WARN:
            feedback.append(
                "⚠ Elbows flaring too wide — tuck elbows 45-60° from torso"
            )

        # ── CHECK 2: Wrist not stacked over elbow ──────────────────
        # If the wrist drifts far in front of or behind the elbow the
        # wrist is carrying a bending moment rather than a pure axial
        # load, risking wrist flexor/extensor strain.
        if l_wrist_drift > self.WRIST_STACK_WARN:
            feedback.append(
                "⚠ Left wrist drifting — stack wrist directly above elbow"
            )
        if r_wrist_drift > self.WRIST_STACK_WARN:
            feedback.append(
                "⚠ Right wrist drifting — stack wrist directly above elbow"
            )

        # ── CHECK 3: Left-right elbow asymmetry ────────────────────
        # A large angle difference between the two elbows indicates
        # one arm is leading the other — a sign of unilateral
        # strength imbalance or uneven bar path.  Over time this
        # can create muscular asymmetry and shoulder impingement on
        # the weaker/lagging side.
        elbow_symmetry_diff = abs(l_elbow_angle - r_elbow_angle)
        if elbow_symmetry_diff > self.SYMMETRY_WARN:
            lagging = "right" if l_elbow_angle < r_elbow_angle else "left"
            feedback.append(
                f"⚠ Uneven press — {lagging} arm lagging, focus on equal drive"
            )

        # ── CHECK 4: Incomplete lockout at the top ─────────────────
        # Catching the rep before full elbow extension forfeits the
        # full concentric range and under-develops the upper pec /
        # anterior deltoid tie-in.  Flag this only at the "up"
        # transition so it doesn't fire during the descent.
        if self.stage == "up" and avg_elbow_angle < self.INCOMPLETE_PRESS_WARN:
            feedback.append(
                "⚠ Incomplete lockout — fully extend arms at the top"
            )

        # ── CHECK 5: Insufficient descent (bar not reaching chest) ──
        # Stopping the eccentric too early (elbow still relatively
        # extended at the "down" trigger) means the athlete is only
        # doing partial reps, missing the deep pec stretch at the
        # bottom and reducing hypertrophic stimulus.
        if self.stage == "down" and avg_elbow_angle > self.INCOMPLETE_LOWER_WARN:
            feedback.append(
                "⚠ Not lowering enough — bring bar to upper chest"
            )

        # ── 5. Return standardised result dict ─────────────────────
        return {
            "rep_count": self.rep_count,
            "stage":     self.stage or "-",
            "feedback":  feedback,
            "angles":    angles,
        }
class DeadliftAnalyser(ExerciseAnalyser):
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    DEADLIFT ANALYSER                             ║
    ║                                                                  ║
    ║  REP CYCLE                                                       ║
    ║  ─────────                                                       ║
    ║  "down"  →  Hip hinge loaded: hip angle < DOWN_THRESH           ║
    ║             (torso parallel-ish to floor, bar at shin level)     ║
    ║  "up"    →  Full lockout:     hip angle > UP_THRESH             ║
    ║             (hips fully extended, glutes squeezed at top)        ║
    ║                                                                  ║
    ║  JOINTS TRACKED                                                  ║
    ║  ──────────────                                                  ║
    ║  • Hip angle    → shoulder – hip – knee   (primary driver)      ║
    ║  • Knee angle   → hip – knee – ankle      (monitors knee bend)  ║
    ║  • Back angle   → vertical ref – mid-hip – mid-shoulder         ║
    ║                   (detects rounding / excessive lean)            ║
    ║  • Shoulder     → tracks bar path proxy                         ║
    ║    over ankle   (shoulder x ≈ ankle x throughout lift)          ║
    ║                                                                  ║
    ║  FORM CHECKS  (5 biomechanical cues)                            ║
    ║  ────────────────────────────────────                            ║
    ║  1. Rounded lower back  (back angle > 55° from vertical)        ║
    ║  2. Hips shooting up    (hip rises faster than shoulders)        ║
    ║  3. Knee cave           (knee x drifts inside ankle x)          ║
    ║  4. Bar drift           (shoulder drifts far forward of ankle)   ║
    ║  5. Incomplete lockout  (hip angle < 160° at "up" transition)   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """

    name = "deadlift"

    # ── Thresholds ─────────────────────────────────────────────────
    # Hip angle (shoulder–hip–knee) in degrees
    DOWN_THRESH = 95    # hip angle ≤ this  →  hinge is loaded ("down")
    UP_THRESH   = 160   # hip angle ≥ this  →  full lockout    ("up")

    # Form-check tolerances
    BACK_ANGLE_WARN    = 55    # degrees from vertical before "rounding" warning
    BAR_DRIFT_WARN     = 0.08  # normalised x-units shoulder drift past ankle
    KNEE_CAVE_WARN     = 0.04  # normalised x-units knee inside ankle
    LOCKOUT_MIN        = 160   # minimum hip angle expected at top of rep

    def __init__(self):
        self.rep_count = 0
        self.stage     = None

        # Used to detect hip-shooting pattern across consecutive frames
        self._prev_hip_y      = None
        self._prev_shoulder_y = None

    def reset(self):
        self.rep_count        = 0
        self.stage            = None
        self._prev_hip_y      = None
        self._prev_shoulder_y = None

    # ──────────────────────────────────────────────────────────────
    def analyse(self, lm) -> dict:
        """
        Parameters
        ----------
        lm : mediapipe NormalizedLandmarkList

        Returns
        -------
        dict with keys: rep_count, stage, feedback, angles
        """
        feedback = []
        angles   = {}

        # ── 1. Gather key coordinates ─────────────────────────────
        l_shoulder = _lm_coords(lm, LM["l_shoulder"])
        r_shoulder = _lm_coords(lm, LM["r_shoulder"])
        l_hip      = _lm_coords(lm, LM["l_hip"])
        r_hip      = _lm_coords(lm, LM["r_hip"])
        l_knee     = _lm_coords(lm, LM["l_knee"])
        r_knee     = _lm_coords(lm, LM["r_knee"])
        l_ankle    = _lm_coords(lm, LM["l_ankle"])
        r_ankle    = _lm_coords(lm, LM["r_ankle"])

        mid_shoulder = midpoint(l_shoulder, r_shoulder)
        mid_hip      = midpoint(l_hip,      r_hip)
        mid_knee     = midpoint(l_knee,     r_knee)
        mid_ankle    = midpoint(l_ankle,    r_ankle)

        # ── 2. Joint angle calculations ───────────────────────────

        # PRIMARY: Hip hinge angle — shoulder–hip–knee
        # Reflects how open/closed the hip hinge is.
        # ~90° = loaded bottom position; ~170°+ = standing lockout
        l_hip_angle = angle_between(l_shoulder, l_hip, l_knee)
        r_hip_angle = angle_between(r_shoulder, r_hip, r_knee)
        avg_hip_angle = (l_hip_angle + r_hip_angle) / 2.0
        angles["Hip"] = avg_hip_angle

        # SECONDARY: Knee angle — hip–knee–ankle
        # In a conventional deadlift the knee has moderate bend (~130-160°
        # at pull initiation); sumo will show more knee bend (~100-120°).
        l_knee_angle = angle_between(l_hip, l_knee, l_ankle)
        r_knee_angle = angle_between(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2.0
        angles["Knee"] = avg_knee_angle

        # TERTIARY: Back (torso) angle from vertical
        # Reference point directly *above* mid_hip in image space
        # (subtract y because image y-axis is inverted — 0 = top).
        vertical_ref = mid_hip + np.array([0.0, -0.5])
        back_angle   = angle_between(vertical_ref, mid_hip, mid_shoulder)
        angles["Back"] = back_angle

        # ── 3. Rep-counting state machine ─────────────────────────
        #
        #   "down" is set when the lifter is in the loaded/hinge position.
        #   A rep is counted only when they return to full lockout ("up")
        #   AFTER having been in the "down" position — preventing false
        #   counts from partial movements.
        #
        if avg_hip_angle <= self.DOWN_THRESH:
            self.stage = "down"

        if avg_hip_angle >= self.UP_THRESH and self.stage == "down":
            self.stage = "up"
            self.rep_count += 1

        # ── 4. Form checks ────────────────────────────────────────

        # ── CHECK 1: Rounded back ──────────────────────────────────
        # The torso should stay as vertical as possible during the pull.
        # Excessive forward lean (> BACK_ANGLE_WARN from vertical) signals
        # a loss of lumbar/thoracic extension — a major injury risk.
        if back_angle > self.BACK_ANGLE_WARN:
            feedback.append("⚠ Back rounding — brace your core & chest up")

        # ── CHECK 2: Bar drift (shoulder over ankle proxy) ─────────
        # In a sound deadlift the bar stays close to the body.  We use the
        # horizontal (x-axis) offset between mid_shoulder and mid_ankle as a
        # proxy: if the shoulders drift far forward of the ankles the lifter
        # is effectively doing a stiff-leg deadlift with a long moment arm,
        # greatly stressing the lower back.
        horizontal_drift = mid_shoulder[0] - mid_ankle[0]
        if horizontal_drift > self.BAR_DRIFT_WARN:
            feedback.append("⚠ Bar drifting forward — keep bar close to body")

        # ── CHECK 3: Knee cave ─────────────────────────────────────
        # Both knees should track in line with (or slightly outside) the
        # corresponding ankle.  Inward collapse increases ACL/MCL stress.
        if l_knee[0] < l_ankle[0] - self.KNEE_CAVE_WARN:
            feedback.append("⚠ Left knee caving — push knee out over toes")
        if r_knee[0] > r_ankle[0] + self.KNEE_CAVE_WARN:
            feedback.append("⚠ Right knee caving — push knee out over toes")

        # ── CHECK 4: Hips shooting up before shoulders ─────────────
        # During the initiation phase ("down" stage), hips and shoulders
        # should rise at roughly the same rate.  If the hips rise
        # significantly faster the torso pitches forward, converting the
        # lift into a back-dominant stiff-leg variation.
        # We detect this by comparing the per-frame delta of hip y vs
        # shoulder y (in image coords, smaller y = higher in frame).
        if self._prev_hip_y is not None and self._prev_shoulder_y is not None:
            hip_rise      = self._prev_hip_y      - mid_hip[1]       # positive = rising
            shoulder_rise = self._prev_shoulder_y - mid_shoulder[1]
            # If hips are rising noticeably faster than shoulders during pull
            if self.stage == "down" and hip_rise - shoulder_rise > 0.015:
                feedback.append("⚠ Hips rising faster than shoulders — drive with legs first")

        # Store current positions for next frame comparison
        self._prev_hip_y      = mid_hip[1]
        self._prev_shoulder_y = mid_shoulder[1]

        # ── CHECK 5: Incomplete lockout at top ─────────────────────
        # At the "up" transition the hips should be fully extended.
        # Catching a rep before true lockout is a common competition fault
        # and also under-develops the glutes.
        if self.stage == "up" and avg_hip_angle < self.LOCKOUT_MIN:
            feedback.append("⚠ Incomplete lockout — fully extend hips & squeeze glutes")

        # ── 5. Return standardised result dict ────────────────────
        return {
            "rep_count": self.rep_count,
            "stage":     self.stage or "-",
            "feedback":  feedback,
            "angles":    angles,
        }
# ──────────────────────────────────────────────────────────────────

EXERCISE_REGISTRY: dict[str, ExerciseAnalyser] = {
    "squat":       SquatAnalyser(),
    "pushup":      PushupAnalyser(),
    "bicep_curl":  BicepCurlAnalyser(),
    "deadlift":    DeadliftAnalyser(),
    "incline_chest_press": InclineChestPressAnalyser(),
    "pullup": PullupAnalyser(),
}


# ──────────────────────────────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────────────────────────────

def load_models(models_dir: str):
    """Load classifier, scaler, and label encoder from disk."""
    required = ["exercise_classifier.pkl", "scaler.pkl", "label_encoder.pkl"]
    for fname in required:
        fpath = os.path.join(models_dir, fname)
        if not os.path.exists(fpath):
            print(f"[ERROR] Missing model file: {fpath}")
            print("        Run  train_model.py  first.")
            sys.exit(1)

    def _load(fname):
        with open(os.path.join(models_dir, fname), "rb") as f:
            return pickle.load(f)

    clf = _load("exercise_classifier.pkl")
    scaler = _load("scaler.pkl")
    le = _load("label_encoder.pkl")
    print(f"[INFO] Loaded classifier: {type(clf).__name__}")
    print(f"[INFO] Exercise classes : {list(le.classes_)}\n")
    return clf, scaler, le


# ──────────────────────────────────────────────────────────────────
# PREDICTION SMOOTHER
# ──────────────────────────────────────────────────────────────────

class PredictionSmoother:
    """
    Keeps a sliding window of the last `window` predictions and returns
    the statistical mode.  Prevents flickering between classes on borderline
    frames.
    """
    def __init__(self, window: int = 10):
        self._buf = collections.deque(maxlen=window)

    def update(self, prediction: str) -> str:
        self._buf.append(prediction)
        try:
            return stat_mode(self._buf)
        except Exception:
            return prediction   # fallback if mode is ambiguous


# ──────────────────────────────────────────────────────────────────
# HUD RENDERING
# ──────────────────────────────────────────────────────────────────

COLOUR = {
    "green":  (0, 210, 90),
    "blue":   (255, 160, 0),
    "red":    (0, 60, 220),
    "yellow": (0, 230, 230),
    "white":  (245, 245, 245),
    "dark":   (20, 20, 30),
    "panel":  (30, 30, 45),
}


def put_text(img, text, pos, scale=0.65, colour=(245, 245, 245),
             thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img, text, pos, font, scale, COLOUR["dark"], thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, colour,          thickness,     cv2.LINE_AA)


def draw_hud(frame: np.ndarray, exercise: str, conf: float,
             result: dict, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]

    # ── Left panel: exercise + rep info ──────────────────────────
    panel_w = 270
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), COLOUR["panel"], -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Exercise label + confidence
    put_text(frame, "EXERCISE", (12, 36), scale=0.5, colour=COLOUR["blue"])
    ex_display = exercise.replace("_", " ").upper() if exercise else "DETECTING…"
    put_text(frame, ex_display, (12, 68), scale=0.85, colour=COLOUR["green"])
    if conf > 0:
        put_text(frame, f"Conf: {conf:.0%}", (12, 94), scale=0.50,
                 colour=COLOUR["yellow"])

    # Stage
    stage_col = COLOUR["green"] if result["stage"] == "up" else COLOUR["yellow"]
    put_text(frame, f"Stage: {result['stage'].upper()}", (12, 130),
             scale=0.6, colour=stage_col)

    # Rep counter — big
    put_text(frame, "REPS", (12, 175), scale=0.5, colour=COLOUR["blue"])
    put_text(frame, str(result["rep_count"]), (12, 250),
             scale=4.0, colour=COLOUR["white"],
             thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX)

    # FPS
    put_text(frame, f"FPS {fps:.0f}", (12, h - 15), scale=0.45,
             colour=(140, 140, 140))

    # ── Bottom bar: angles ────────────────────────────────────────
    bar_h = 32
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (panel_w, h - bar_h), (w, h), COLOUR["dark"], -1)
    cv2.addWeighted(overlay2, 0.70, frame, 0.30, 0, frame)
    x_cursor = panel_w + 10
    for name, val in result["angles"].items():
        angle_str = f"{name}: {val:.0f}°"
        put_text(frame, angle_str, (x_cursor, h - 10), scale=0.50,
                 colour=COLOUR["yellow"])
        x_cursor += 130

    # ── Right side: form feedback ─────────────────────────────────
    if result["feedback"]:
        fb_x = w - 390
        for i, msg in enumerate(result["feedback"][:4]):
            y = 40 + i * 30
            put_text(frame, msg, (fb_x, y), scale=0.52,
                     colour=COLOUR["red"])
    else:
        put_text(frame, "✓ Form looks good!", (w - 260, 40),
                 scale=0.58, colour=COLOUR["green"])

    return frame


# ──────────────────────────────────────────────────────────────────
# MAIN TRACKER LOOP
# ──────────────────────────────────────────────────────────────────

def run_tracker(args) -> None:
    # ── Load ML artefacts ─────────────────────────────────────────
    if args.force_exercise:
        clf = scaler = le = None
        forced_exercise = args.force_exercise.lower().strip()
        print(f"[INFO] Forced exercise mode: {forced_exercise}")
        if forced_exercise not in EXERCISE_REGISTRY:
            print(f"[ERROR] '{forced_exercise}' not in registry: {list(EXERCISE_REGISTRY.keys())}")
            sys.exit(1)
    else:
        clf, scaler, le = load_models(args.models)
        forced_exercise = None

    # ── MediaPipe setup ───────────────────────────────────────────
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    # ── Video I/O ─────────────────────────────────────────────────
    cap_source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional video writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 30, (width, height))
        print(f"[INFO] Saving annotated output → {args.save}")

    smoother        = PredictionSmoother(window=args.smooth)
    current_analyser: Optional[ExerciseAnalyser] = None
    last_exercise   = None
    prev_time       = time.time()
    conf            = 0.0

    # ── Null result for frames with no detection ──────────────────
    null_result = {"rep_count": 0, "stage": "-", "feedback": [], "angles": {}}

    print("[INFO] Tracker running.  Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of source reached.")
            break

        # FPS
        cur_time = time.time()
        fps = 1.0 / max(cur_time - prev_time, 1e-6)
        prev_time = cur_time

        # ── Pose detection ────────────────────────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # Draw skeleton
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 210, 255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1),
            )

        # ── Classification ────────────────────────────────────────
        exercise = forced_exercise
        result   = null_result

        if results.pose_landmarks:
            feats = extract_features(results.pose_landmarks)
            if feats is not None and clf is not None:
                feats_scaled = scaler.transform(feats)
                raw_pred     = le.inverse_transform(clf.predict(feats_scaled))[0]
                smoothed     = smoother.update(raw_pred)
                exercise     = smoothed

                # Confidence (predict_proba if available)
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(feats_scaled)[0]
                    conf  = float(proba.max())
                else:
                    conf = 1.0
            elif forced_exercise:
                exercise = forced_exercise

            # ── Route to correct analyser ─────────────────────────
            if exercise and exercise in EXERCISE_REGISTRY:
                if exercise != last_exercise:
                    # Exercise changed — reset the analyser's state
                    EXERCISE_REGISTRY[exercise].reset()
                    last_exercise = exercise
                    print(f"[INFO] Exercise detected: {exercise.upper()}")

                current_analyser = EXERCISE_REGISTRY[exercise]
                result = current_analyser.analyse(results.pose_landmarks)

        # ── HUD ───────────────────────────────────────────────────
        frame = draw_hud(frame, exercise or "", conf, result, fps)

        if writer:
            writer.write(frame)

        cv2.imshow("AI Fitness Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    pose.close()

    # Final rep summary
    print("\n" + "═" * 45)
    print("  SESSION SUMMARY")
    print("═" * 45)
    for ex_name, analyser in EXERCISE_REGISTRY.items():
        if analyser.rep_count > 0:
            print(f"  {ex_name:20s}  {analyser.rep_count:3d} reps")
    print("═" * 45 + "\n")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time AI fitness tracker: exercise classification + rep counting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source",          default="0",
                   help="Webcam index or path to video file  [default: 0]")
    p.add_argument("--models",          default="models",
                   help="Directory containing trained model artefacts  [default: models/]")
    p.add_argument("--smooth",          type=int, default=10,
                   help="Prediction smoothing window (frames)  [default: 10]")
    p.add_argument("--save",            default=None,
                   help="Optional path to save annotated output video  (e.g. output.mp4)")
    p.add_argument("--force-exercise",  default=None,
                   dest="force_exercise",
                   help="Skip ML classification and force a specific exercise label")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tracker(args)
