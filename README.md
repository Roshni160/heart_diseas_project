# Heart Disease Risk - EDA & Classification with SHAP

**Domain:** Healthcare / Medical  
**Tools:** Python · scikit-learn · SHAP · Pandas · Seaborn · Matplotlib

---

## Problem Statement

Heart disease is the leading cause of death globally. Early identification of at risk patients enables preventive intervention. This project uses the **UCI Heart Disease dataset** (303 patients, 13 clinical features) to:

1. Explore which patient features correlate with heart disease
2. Build a Random Forest classifier to predict disease presence
3. Use **SHAP** to explain *why* the model makes each prediction

---

## Dataset

| Detail | Value |
|---|---|
| Source | [UCI ML Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Patients | 303 |
| Features | 13 clinical + 1 target |
| Target | Binary (0 = no disease, 1 = disease present) |
| Missing values | None (after preprocessing) |

### Feature Dictionary

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dL) |
| fbs | Fasting blood sugar > 120 mg/dL (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels coloured by fluoroscopy (0–3) |
| thal | Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversable defect) |

---

## Methodology

```
Raw Data → EDA → Feature Analysis → Train/Test Split → Random Forest → SHAP Explainability
```

### 1. EDA
- Class distribution check
- Histograms split by disease status (5 numeric features)
- Correlation heatmap (lower triangle)
- Categorical feature grouped bar charts

### 2. Modelling
- **Algorithm:** Random Forest (n=300 trees, balanced class weights)
- **Validation:** 5 fold stratified cross-validation
- **Metrics:** ROC-AUC, precision, recall, F1

### 3. Explainability (SHAP)
- Beeswarm summary plot - direction + magnitude per feature
- Bar chart - mean absolute SHAP (global importance)
- Dependence plots - how top features interact
- Force plot - single high-risk patient explanation

---

## Results

| Metric | Score |
|---|---|
| 5-Fold CV ROC-AUC | ~0.90 ± 0.03 |
| Test ROC-AUC | ~0.91 |
| Test Accuracy | ~85% |

### Key SHAP Findings

| Feature | Finding |
|---|---|
| Chest Pain Type | Asymptomatic patients (cp=3) are paradoxically highest risk |
| Max Heart Rate | Lower values strongly push prediction toward disease |
| ST Depression | Near-linear SHAP increase - direct ischaemia marker |
| Major Vessels (ca) | More vessels = stronger disease signal |
| Thalassemia | Reversible defect is the single strongest predictor |

---

## Output Files

| File | Description |
|---|---|
| `01_class_distribution.png` | Target class balance |
| `02_distributions.png` | Feature distributions by disease status |
| `03_correlation_heatmap.png` | Full correlation matrix |
| `04_categorical_features.png` | Disease rate by categorical feature |
| `05_confusion_roc.png` | Confusion matrix + ROC curve |
| `06_feature_importance.png` | RF built-in feature importances |
| `07_shap_summary.png` | SHAP beeswarm plot |
| `08_shap_bar.png` | Mean absolute SHAP values |
| `09_shap_dependence.png` | Dependence plots for top 2 features |
| `10_shap_force.png` | Force plot for a high-risk patient |

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/roshni-makwana/heart-disease-risk
cd heart-disease-risk

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run as Python script
python heart_disease_analysis.py

# 3b. Or open the notebook
jupyter notebook heart_disease_analysis.ipynb
```

---

## Dependencies

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
shap>=0.42
ucimlrepo>=0.0.3
```

---

## Learnings & Next Steps

- **SHAP makes ML transparent** - interpretability is critical in healthcare where trust matters
- **Cholesterol was not the top predictor** - ST depression and max heart rate were more informative
- **Next steps:** Try LightGBM or XGBoost, add LIME comparison, deploy as a Streamlit risk calculator

---
