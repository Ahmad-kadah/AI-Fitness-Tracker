"""
╔══════════════════════════════════════════════════════════════════╗
║           STAGE 2 — MACHINE LEARNING CLASSIFIER TRAINER          ║
║  Trains a Random Forest (default) or SVM on the landmark CSV     ║
║  produced by data_collector.py, then saves the model + scaler.   ║
╚══════════════════════════════════════════════════════════════════╝

USAGE
─────
  # Train with default Random Forest:
  python train_model.py --data data/landmarks.csv --output models/

  # Train with SVM:
  python train_model.py --data data/landmarks.csv --output models/ --model svm

  # Show full classification report only (no retrain):
  python train_model.py --data data/landmarks.csv --output models/ --eval-only

OUTPUT FILES
────────────
  models/exercise_classifier.pkl   ← trained classifier
  models/scaler.pkl                ← fitted StandardScaler
  models/label_encoder.pkl         ← fitted LabelEncoder
  models/training_report.txt       ← human-readable metrics
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Optional: matplotlib for confusion matrix plot (gracefully skipped if absent)
try:
    import matplotlib
    matplotlib.use("Agg")          # headless — no display needed
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str):
    """
    Load the landmark CSV and split into feature matrix X and label vector y.
    Drops rows where any landmark value is NaN.
    """
    print(f"[INFO] Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        print("[ERROR] CSV is missing a 'label' column.  Re-run data_collector.py.")
        sys.exit(1)

    df.dropna(inplace=True)

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"[INFO] Dataset shape : {X.shape}")
    print(f"[INFO] Classes found : {list(le.classes_)}")
    for cls in le.classes_:
        n = (y_raw == cls).sum()
        print(f"         {cls:20s}  {n:5d} samples")
    print()
    return X, y, le


def build_model(model_type: str, n_classes: int):
    """
    Return a scikit-learn estimator (NOT wrapped in a pipeline; scaler is
    handled separately so we can save it independently for inference).
    """
    model_type = model_type.lower()
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
    elif model_type == "svm":
        clf = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,          # needed for predict_proba in main_tracker
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
    else:
        print(f"[ERROR] Unknown model type '{model_type}'. Choose: rf | svm | gb")
        sys.exit(1)

    return clf


def save_artifact(obj, path: str) -> None:
    """Pickle an artifact to disk."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVE] {path}")


def plot_confusion_matrix(cm, class_names: list, out_path: str) -> None:
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(max(4, len(class_names)), max(4, len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Exercise Classifier", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVE] {out_path}")


# ──────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────

def train(args) -> None:
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────
    X, y, le = load_dataset(args.data)

    if len(le.classes_) < 2:
        print("[ERROR] Need at least 2 exercise classes in the dataset.")
        sys.exit(1)

    # ── 2. Scale features ─────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 3. Train / test split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, stratify=y, random_state=42
    )

    # ── 4. Cross-validation (on training set) ─────────────────────
    clf = build_model(args.model, n_classes=len(le.classes_))
    print(f"[INFO] Running 5-fold cross-validation with {args.model.upper()} …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    t0 = time.time()
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv,
                                scoring="accuracy", n_jobs=-1)
    print(f"[INFO] CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"[INFO] CV took {time.time() - t0:.1f}s\n")

    # ── 5. Final fit on full training set ─────────────────────────
    print("[INFO] Fitting final model on full training split …")
    clf.fit(X_train, y_train)

    # ── 6. Evaluate on hold-out test set ─────────────────────────
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_test, y_pred)

    print("─" * 52)
    print("CLASSIFICATION REPORT (20% hold-out)")
    print("─" * 52)
    print(report)

    # ── 7. Save artefacts ─────────────────────────────────────────
    out = args.output.rstrip("/\\")
    save_artifact(clf,    f"{out}/exercise_classifier.pkl")
    save_artifact(scaler, f"{out}/scaler.pkl")
    save_artifact(le,     f"{out}/label_encoder.pkl")

    # Persist the training report as plain text
    report_path = f"{out}/training_report.txt"
    with open(report_path, "w") as rp:
        rp.write("AI FITNESS TRACKER — TRAINING REPORT\n")
        rp.write("=" * 52 + "\n")
        rp.write(f"Model      : {args.model.upper()}\n")
        rp.write(f"Dataset    : {args.data}\n")
        rp.write(f"Classes    : {list(le.classes_)}\n")
        rp.write(f"Samples    : {len(X)}\n")
        rp.write(f"CV Score   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        rp.write("CLASSIFICATION REPORT (20% hold-out)\n")
        rp.write("-" * 52 + "\n")
        rp.write(report + "\n")
        rp.write("CONFUSION MATRIX\n")
        rp.write("-" * 52 + "\n")
        rp.write(str(cm) + "\n")
    print(f"[SAVE] {report_path}")

    # Optional confusion matrix plot
    cm_plot_path = f"{out}/confusion_matrix.png"
    plot_confusion_matrix(cm, list(le.classes_), cm_plot_path)

    print("\n[DONE] Training complete.  All artefacts saved to:", out)
    print("       Run  main_tracker.py  to start real-time inference.\n")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train an exercise classifier on MediaPipe landmark features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data",   default="data/landmarks.csv",
                   help="Path to the landmark CSV from data_collector.py  [default: data/landmarks.csv]")
    p.add_argument("--output", default="models",
                   help="Directory to store model artefacts  [default: models/]")
    p.add_argument("--model",  default="rf", choices=["rf", "svm", "gb"],
                   help="Classifier type: rf (Random Forest) | svm | gb (Gradient Boost)  [default: rf]")
    p.add_argument("--eval-only", action="store_true",
                   help="Only print metrics; skip saving artefacts (useful for quick checks)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
