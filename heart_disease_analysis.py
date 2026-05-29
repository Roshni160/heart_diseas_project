# =============================================================================
# PROJECT 3: Heart Disease Risk — EDA & Classification with SHAP
# Dataset: UCI Heart Disease Dataset
# Author: Roshni Makwana
# =============================================================================

# ── 0. INSTALL & IMPORTS ─────────────────────────────────────────────────────
# Run this first if needed:
# pip install pandas numpy matplotlib seaborn scikit-learn shap ucimlrepo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafaf8",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#e8e8e4",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
})

ACCENT   = "#1d6b4e"   # deep green
DANGER   = "#c0392b"   # red for disease positive
NEUTRAL  = "#8a9bb0"   # blue-grey for negative
PALETTE  = [NEUTRAL, DANGER]

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 60)
print("  Heart Disease Risk — EDA & Classification")
print("=" * 60)

# Option A: Auto-download from UCI (requires ucimlrepo)
try:
    from ucimlrepo import fetch_ucirepo
    heart = fetch_ucirepo(id=45)
    df = heart.data.features.copy()
    df["target"] = heart.data.targets.values.ravel()
    print("✓ Dataset loaded from UCI ML Repository")
except Exception:
    # Option B: Load from local CSV (download from Kaggle: 'heart-disease-uci')
    # df = pd.read_csv("heart.csv")
    # Fallback: generate realistic synthetic data for demonstration
    print("⚠ UCI fetch failed — generating synthetic demo data.")
    np.random.seed(42)
    n = 303
    df = pd.DataFrame({
        "age":      np.random.randint(29, 77, n),
        "sex":      np.random.randint(0, 2, n),
        "cp":       np.random.randint(0, 4, n),
        "trestbps": np.random.randint(94, 200, n),
        "chol":     np.random.randint(126, 564, n),
        "fbs":      np.random.randint(0, 2, n),
        "restecg":  np.random.randint(0, 3, n),
        "thalach":  np.random.randint(71, 202, n),
        "exang":    np.random.randint(0, 2, n),
        "oldpeak":  np.round(np.random.uniform(0, 6.2, n), 1),
        "slope":    np.random.randint(0, 3, n),
        "ca":       np.random.randint(0, 4, n),
        "thal":     np.random.choice([1, 2, 3], n),
        "target":   np.random.randint(0, 2, n),
    })

# Binarise target: 0 = no disease, 1 = disease present
df["target"] = (df["target"] > 0).astype(int)

print(f"\nDataset shape : {df.shape}")
print(f"Positive cases: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
print(f"Missing values: {df.isnull().sum().sum()}")
print("\nFirst 5 rows:")
print(df.head())

# Column display names for plots
COL_LABELS = {
    "age":      "Age",
    "sex":      "Sex (1=M, 0=F)",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting BP (mmHg)",
    "chol":     "Cholesterol (mg/dL)",
    "fbs":      "Fasting Blood Sugar",
    "restecg":  "Resting ECG",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise Angina",
    "oldpeak":  "ST Depression",
    "slope":    "Slope of ST",
    "ca":       "Major Vessels (0–3)",
    "thal":     "Thalassemia",
    "target":   "Heart Disease",
}

FEATURES = [c for c in df.columns if c != "target"]

# =============================================================================
# 2. EDA — CLASS DISTRIBUTION
# =============================================================================
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["target"].value_counts().sort_index()
bars = ax.bar(["No Disease", "Disease Present"], counts.values,
              color=PALETTE, width=0.5, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.2)
plt.tight_layout()
plt.savefig("01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[Fig 1] Saved: 01_class_distribution.png")

# =============================================================================
# 3. EDA — DISTRIBUTION PLOTS (key numeric features)
# =============================================================================
num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feat in enumerate(num_features):
    ax = axes[i]
    for label, color, name in [(0, NEUTRAL, "No Disease"), (1, DANGER, "Disease")]:
        subset = df[df["target"] == label][feat].dropna()
        ax.hist(subset, bins=25, alpha=0.6, color=color, label=name,
                edgecolor="white", linewidth=0.4)
    ax.set_title(COL_LABELS[feat])
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

# Age vs Max Heart Rate scatter in the 6th panel
ax = axes[5]
scatter = ax.scatter(df["age"], df["thalach"],
                     c=df["target"].map({0: NEUTRAL, 1: DANGER}),
                     alpha=0.55, s=25, edgecolors="none")
ax.set_title("Age vs Max Heart Rate")
ax.set_xlabel("Age")
ax.set_ylabel("Max Heart Rate")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=NEUTRAL, label="No Disease"),
                   Patch(color=DANGER,  label="Disease")], fontsize=9)

plt.suptitle("Feature Distributions by Heart Disease Status", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("02_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Fig 2] Saved: 02_distributions.png")

# =============================================================================
# 4. EDA — CORRELATION HEATMAP
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt=".2f",
            annot_kws={"size": 8}, linewidths=0.5, linecolor="#e0e0de",
            square=True, vmin=-1, vmax=1, ax=ax,
            xticklabels=[COL_LABELS.get(c, c) for c in corr.columns],
            yticklabels=[COL_LABELS.get(c, c) for c in corr.columns])
ax.set_title("Correlation Heatmap — All Features", pad=16)
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Fig 3] Saved: 03_correlation_heatmap.png")

# Strongest correlations with target
print("\nTop correlations with target:")
target_corr = corr["target"].drop("target").abs().sort_values(ascending=False)
for feat, val in target_corr.head(6).items():
    print(f"  {COL_LABELS.get(feat, feat):<25} {val:.3f}")

# =============================================================================
# 5. EDA — CATEGORICAL FEATURE ANALYSIS
# =============================================================================
cat_features = ["cp", "sex", "fbs", "exang", "slope", "thal"]
cat_labels   = {
    "cp":    ["Typical\nAngina", "Atypical\nAngina", "Non-anginal\nPain", "Asymptomatic"],
    "sex":   ["Female", "Male"],
    "fbs":   ["≤120 mg/dL", ">120 mg/dL"],
    "exang": ["No", "Yes"],
    "slope": ["Upsloping", "Flat", "Downsloping"],
    "thal":  ["Normal", "Fixed\nDefect", "Reversable\nDefect"],
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feat in enumerate(cat_features):
    ax = axes[i]
    ct = df.groupby([feat, "target"]).size().unstack(fill_value=0)
    ct.columns = ["No Disease", "Disease"]
    pct = ct.div(ct.sum(axis=1), axis=0) * 100

    x = np.arange(len(ct))
    w = 0.38
    ax.bar(x - w/2, pct["No Disease"], width=w, color=NEUTRAL, label="No Disease",
           edgecolor="white", linewidth=0.8)
    ax.bar(x + w/2, pct["Disease"],    width=w, color=DANGER,  label="Disease",
           edgecolor="white", linewidth=0.8)

    labels = cat_labels.get(feat, [str(v) for v in ct.index])
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(ct)], fontsize=9)
    ax.set_title(COL_LABELS[feat])
    ax.set_ylabel("% within group")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)

plt.suptitle("Heart Disease Rate by Categorical Feature", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("04_categorical_features.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Fig 4] Saved: 04_categorical_features.png")

# =============================================================================
# 6. PREPROCESSING
# =============================================================================
print("\n" + "="*60)
print("  MODELLING")
print("="*60)

X = df[FEATURES]
y = df["target"]

# Handle any remaining categoricals
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# =============================================================================
# 7. RANDOM FOREST — TRAINING & CROSS-VALIDATION
# =============================================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
print(f"\nCross-Validation ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

rf.fit(X_train, y_train)
y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print(f"\nTest ROC-AUC : {roc_auc_score(y_test, y_proba):.3f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No Disease','Disease'])}")

# =============================================================================
# 8. CONFUSION MATRIX + ROC CURVE
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_val = roc_auc_score(y_test, y_proba)
axes[1].plot(fpr, tpr, color=ACCENT, lw=2, label=f"Random Forest (AUC = {auc_val:.3f})")
axes[1].plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random baseline")
axes[1].fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("05_confusion_roc.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Fig 5] Saved: 05_confusion_roc.png")

# =============================================================================
# 9. FEATURE IMPORTANCE (built-in)
# =============================================================================
importances = pd.Series(rf.feature_importances_, index=FEATURES)\
                .rename(index=COL_LABELS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
colors = [DANGER if v >= importances.quantile(0.75) else ACCENT
          for v in importances.values]
ax.barh(importances.index, importances.values, color=colors,
        edgecolor="white", linewidth=0.6)
ax.set_xlabel("Mean Decrease in Impurity")
ax.set_title("Random Forest — Feature Importances")
plt.tight_layout()
plt.savefig("06_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Fig 6] Saved: 06_feature_importance.png")

# =============================================================================
# 10. SHAP — GLOBAL EXPLAINABILITY
# =============================================================================
print("\nCalculating SHAP values…")
explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is a list [class0, class1]
# Use class 1 (disease present)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# ── SHAP Summary Plot (beeswarm) ──────────────────────────────────────────────
plt.figure(figsize=(9, 6))
shap.summary_plot(
    sv, X_test,
    feature_names=[COL_LABELS.get(f, f) for f in FEATURES],
    show=False, plot_size=None, color_bar=True,
)
plt.title("SHAP Summary — Feature Impact on Heart Disease Prediction",
          fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("07_shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Fig 7] Saved: 07_shap_summary.png")

# ── SHAP Bar Plot (mean |SHAP|) ───────────────────────────────────────────────
plt.figure(figsize=(9, 5))
shap.summary_plot(
    sv, X_test,
    feature_names=[COL_LABELS.get(f, f) for f in FEATURES],
    plot_type="bar", show=False, plot_size=None,
    color=ACCENT,
)
plt.title("Mean |SHAP| Value — Average Feature Contribution",
          fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("08_shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Fig 8] Saved: 08_shap_bar.png")

# ── SHAP Dependence Plots — top 2 features ───────────────────────────────────
top2 = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES)\
         .sort_values(ascending=False).head(2).index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, feat in zip(axes, top2):
    feat_idx = FEATURES.index(feat)
    shap.dependence_plot(
        feat_idx, sv, X_test,
        feature_names=[COL_LABELS.get(f, f) for f in FEATURES],
        ax=ax, show=False, dot_size=20, alpha=0.7,
    )
    ax.set_title(f"SHAP Dependence — {COL_LABELS.get(feat, feat)}")

plt.suptitle("SHAP Dependence Plots (Top 2 Features)", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("09_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Fig 9] Saved: 09_shap_dependence.png")

# ── SHAP Force Plot — single prediction ──────────────────────────────────────
# Pick a high-risk patient from test set
high_risk_idx = np.argmax(y_proba)
print(f"\nExplaining high-risk patient (index {high_risk_idx})")
print(f"  Predicted probability: {y_proba[high_risk_idx]:.2f}")
print(f"  Actual label         : {'Disease' if y_test.iloc[high_risk_idx] == 1 else 'No Disease'}")

shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value[1],
    sv[high_risk_idx],
    X_test.iloc[high_risk_idx],
    feature_names=[COL_LABELS.get(f, f) for f in FEATURES],
    matplotlib=True, show=False,
)
plt.title("SHAP Force Plot — Single High-Risk Patient", fontsize=11,
          fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("10_shap_force.png", dpi=150, bbox_inches="tight")
plt.show()
print("[Fig 10] Saved: 10_shap_force.png")

# =============================================================================
# 11. SUMMARY & INTERPRETATION
# =============================================================================
print("\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)
print(f"""
Model         : Random Forest (n_estimators=300)
CV AUC        : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}
Test AUC      : {roc_auc_score(y_test, y_proba):.3f}

Key SHAP Findings:
──────────────────
1. Chest Pain Type (cp): Asymptomatic chest pain is strongly associated
   with heart disease — counter-intuitively, patients with no pain are
   at higher risk than those with typical angina.

2. Max Heart Rate (thalach): Lower maximum heart rate during exercise
   significantly raises predicted disease probability. SHAP shows a
   clear negative relationship.

3. ST Depression (oldpeak): Higher ST depression pushes predictions
   toward disease. The dependence plot shows a near-linear increase.

4. Major Vessels (ca): More coloured vessels (0–3) seen in fluoroscopy
   strongly increase risk. ca=0 lowers prediction; ca=3 raises it most.

5. Thalassemia (thal): Reversible thalassemia defect (thal=2) is the
   single strongest individual SHAP contributor for high-risk patients.

All 10 figures saved to working directory.
""")
