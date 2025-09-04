# main.py — Diabetes prediction with XGBoost + threshold tuning + figures
# Runs on Stampede3 or locally. Saves metrics and plots to results/

from pathlib import Path
import json, datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend for HPC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from xgboost import XGBClassifier

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[1]   # repo root (…/Diabetes-Prediction)
DATA = BASE_DIR / "data" / "diabetes_prediction_dataset.csv"
RESULTS = BASE_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ---------- Load data ----------
df = pd.read_csv(DATA)

# ---------- Cleaning & encoding (matches README) ----------
df = df.drop_duplicates()
df = df[(df["age"] > 0) & (df["age"] <= 120)]
df = df[df["bmi"] <= 60]
df = df[(df["HbA1c_level"] >= 3) & (df["HbA1c_level"] <= 20)]
df = df[(df["blood_glucose_level"] >= 50) & (df["blood_glucose_level"] <= 400)]

# smoking history: collapse categories and encode
df["smoking_history"] = df["smoking_history"].replace({"not current": "former", "ever": "former"})
smoking_map = {"never": 0, "former": 1, "current": 2}
df["smoking_history_encoded"] = df["smoking_history"].map(smoking_map).fillna(4).astype(int)

# gender: keep Female/Male and encode
df = df[df["gender"].isin(["Female", "Male"])]
gender_map = {"Female": 0, "Male": 1}
df["gender_encoded"] = df["gender"].map(gender_map).astype(int)

# features/target
FEATURES = [
    "gender_encoded", "age", "hypertension", "heart_disease",
    "smoking_history_encoded", "bmi", "HbA1c_level", "blood_glucose_level"
]
X = df[FEATURES]
y = df["diabetes"].astype(int)

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------- Model ----------
xgb_model = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
    random_state=42,
)
xgb_model.fit(X_train, y_train)

# ---------- Predict probabilities ----------
y_probs = xgb_model.predict_proba(X_test)[:, 1]

# ---------- Threshold search: aim for recall >= 0.80, maximize F1 ----------
thresholds = np.linspace(0.10, 0.50, 81)
best_threshold = None
best_f1 = -1.0
best_report = None
best_preds = None

for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    rpt = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    recall_pos = rpt["1"]["recall"]
    f1_pos = rpt["1"]["f1-score"]
    if recall_pos >= 0.80 and f1_pos > best_f1:
        best_f1 = f1_pos
        best_threshold = float(t)
        best_report = rpt
        best_preds = y_pred

# fallback: if nothing reached 0.80 recall, pick best F1 overall
if best_threshold is None:
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        rpt = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1_pos = rpt["1"]["f1-score"]
        if f1_pos > best_f1:
            best_f1 = f1_pos
            best_threshold = float(t)
            best_report = rpt
            best_preds = y_pred

# ---------- Metrics ----------
cm = confusion_matrix(y_test, best_preds)
roc_auc = roc_auc_score(y_test, y_probs)

print(f"Chosen threshold with recall target (>=0.80 if possible): {best_threshold:.6f}")
print(classification_report(y_test, best_preds, zero_division=0))
print(cm)
print("ROC-AUC Score:", roc_auc)

# ---------- Plots ----------
# ROC
RocCurveDisplay.from_predictions(y_test, y_probs)
plt.title("ROC Curve")
plt.savefig(RESULTS / "roc_curve.png", dpi=200, bbox_inches="tight")
plt.clf()

# Precision–Recall
PrecisionRecallDisplay.from_predictions(y_test, y_probs)
plt.title("Precision–Recall Curve")
plt.savefig(RESULTS / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
plt.clf()

# Confusion Matrix heatmap
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_xticks([0, 1]); ax.set_xticklabels(["0", "1"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["0", "1"])
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", color="black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.clf()

# Feature importance
importances = xgb_model.feature_importances_
plt.bar(range(len(importances)), importances)
plt.title("XGBoost Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.xticks(range(len(importances)), range(len(importances)))
plt.tight_layout()
plt.savefig(RESULTS / "feature_importance.png", dpi=200, bbox_inches="tight")
plt.clf()

# ---------- Run metadata ----------
meta = {
    "threshold": best_threshold,
    "precision_pos": best_report["1"]["precision"],
    "recall_pos": best_report["1"]["recall"],
    "f1_pos": best_report["1"]["f1-score"],
    "accuracy": best_report["accuracy"],
    "roc_auc": float(roc_auc),
    "random_state": 42,
    "features": FEATURES,
    "data_path": str(DATA),
    "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
}
with open(RESULTS / "run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Saved plots and metadata to: {RESULTS}")
# main.py — Diabetes prediction with XGBoost + threshold tuning + figures
# Runs on Stampede3 or locally. Saves metrics and plots to results/

from pathlib import Path
import json, datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend for HPC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from xgboost import XGBClassifier

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[1]   # repo root (…/Diabetes-Prediction)
DATA = BASE_DIR / "data" / "diabetes_prediction_dataset.csv"
RESULTS = BASE_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ---------- Load data ----------
df = pd.read_csv(DATA)

# ---------- Cleaning & encoding (matches README) ----------
df = df.drop_duplicates()
df = df[(df["age"] > 0) & (df["age"] <= 120)]
df = df[df["bmi"] <= 60]
df = df[(df["HbA1c_level"] >= 3) & (df["HbA1c_level"] <= 20)]
df = df[(df["blood_glucose_level"] >= 50) & (df["blood_glucose_level"] <= 400)]

# smoking history: collapse categories and encode
df["smoking_history"] = df["smoking_history"].replace({"not current": "former", "ever": "former"})
smoking_map = {"never": 0, "former": 1, "current": 2}
df["smoking_history_encoded"] = df["smoking_history"].map(smoking_map).fillna(4).astype(int)

# gender: keep Female/Male and encode
df = df[df["gender"].isin(["Female", "Male"])]
gender_map = {"Female": 0, "Male": 1}
df["gender_encoded"] = df["gender"].map(gender_map).astype(int)

# features/target
FEATURES = [
    "gender_encoded", "age", "hypertension", "heart_disease",
    "smoking_history_encoded", "bmi", "HbA1c_level", "blood_glucose_level"
]
X = df[FEATURES]
y = df["diabetes"].astype(int)

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------- Model ----------
xgb_model = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
    random_state=42,
)
xgb_model.fit(X_train, y_train)

# ---------- Predict probabilities ----------
y_probs = xgb_model.predict_proba(X_test)[:, 1]

# ---------- Threshold search: aim for recall >= 0.80, maximize F1 ----------
thresholds = np.linspace(0.10, 0.50, 81)
best_threshold = None
best_f1 = -1.0
best_report = None
best_preds = None

for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    rpt = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    recall_pos = rpt["1"]["recall"]
    f1_pos = rpt["1"]["f1-score"]
    if recall_pos >= 0.80 and f1_pos > best_f1:
        best_f1 = f1_pos
        best_threshold = float(t)
        best_report = rpt
        best_preds = y_pred

# fallback: if nothing reached 0.80 recall, pick best F1 overall
if best_threshold is None:
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        rpt = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1_pos = rpt["1"]["f1-score"]
        if f1_pos > best_f1:
            best_f1 = f1_pos
            best_threshold = float(t)
            best_report = rpt
            best_preds = y_pred

# ---------- Metrics ----------
cm = confusion_matrix(y_test, best_preds)
roc_auc = roc_auc_score(y_test, y_probs)

print(f"Chosen threshold with recall target (>=0.80 if possible): {best_threshold:.6f}")
print(classification_report(y_test, best_preds, zero_division=0))
print(cm)
print("ROC-AUC Score:", roc_auc)

# ---------- Plots ----------
# ROC
RocCurveDisplay.from_predictions(y_test, y_probs)
plt.title("ROC Curve")
plt.savefig(RESULTS / "roc_curve.png", dpi=200, bbox_inches="tight")
plt.clf()

# Precision–Recall
PrecisionRecallDisplay.from_predictions(y_test, y_probs)
plt.title("Precision–Recall Curve")
plt.savefig(RESULTS / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
plt.clf()

# Confusion Matrix heatmap
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_xticks([0, 1]); ax.set_xticklabels(["0", "1"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["0", "1"])
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", color="black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.clf()

importances = xgb_model.feature_importances_
plt.bar(range(len(importances)), importances)
plt.title("XGBoost Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.xticks(range(len(importances)), range(len(importances)))
plt.tight_layout()
plt.savefig(RESULTS / "feature_importance.png", dpi=200, bbox_inches="tight")
plt.clf()

meta = {
    "threshold": best_threshold,
    "precision_pos": best_report["1"]["precision"],
    "recall_pos": best_report["1"]["recall"],
    "f1_pos": best_report["1"]["f1-score"],
    "accuracy": best_report["accuracy"],
    "roc_auc": float(roc_auc),
    "random_state": 42,
    "features": FEATURES,
    "data_path": str(DATA),
    "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
}
with open(RESULTS / "run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Saved plots and metadata to: {RESULTS}")
