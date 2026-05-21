# 🧠 Alzheimer's Disease Prediction and Comprehensive Data Analysis Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-4fd1c5?style=flat-square)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-96.74%25-brightgreen?style=flat-square)

> **Final Project** — Big Data Analytics Course, December 2025  
> Department of Computer Science and Information Engineering, NDHU  
> Advised by **Prof. Chung Yung** & TA **Demiah Kisla Charlery**  
> **Author:** [Supharada Jundok](https://mato8q.github.io/imnotmato8q.github.io/)

---

## 📌 Overview

This project applies machine learning techniques to predict Alzheimer's disease diagnosis from structured clinical data. The analysis covers the full pipeline — from exploratory data analysis and feature selection to multi-model comparison, evaluation, and SHAP-based explainability.

**Key result:** Random Forest achieved **96.74% test accuracy** and **97.48% ROC-AUC**, outperforming 10 other classifiers.

🌐 **Live Showcase:** [View Project Website](https://mato8q.github.io/Alzheimer_s_Disease_PredictionNComprehensive_Data_Analysis_Using_ML)

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Alzheimer's Disease Dataset — Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) |
| Total samples | 2,149 patients |
| Non-Alzheimer (0) | 1,389 samples (64.6%) |
| Alzheimer (1) | 760 samples (35.4%) |
| Total features | 35 columns |
| Selected features | 5 (post-EDA) |

**Class imbalance** was handled via `class_weight='balanced'` for tree-based models and `scale_pos_weight` for XGBoost/LightGBM.

---

## 🔬 Selected Features

Chosen through correlation analysis, t-tests, chi-square tests, and Spearman correlation:

| Feature | Type | Gini Importance | Reason |
|---|---|---|---|
| `FunctionalAssessment` | Continuous | 0.35 | Highest discriminative power |
| `ADL` | Continuous | 0.22 | Strong separation between classes |
| `MMSE` | Continuous | 0.15 | Core clinical cognitive screening |
| `MemoryComplaints` | Binary | 0.08 | Significant chi-square (p < 0.05) |
| `BehavioralProblems` | Binary | 0.05 | Significant chi-square (p < 0.05) |

---

## ⚙️ Project Pipeline

```
Dataset Acquisition
      ↓
Data Exploration & Cleaning   (missing values, distributions)
      ↓
Exploratory Data Analysis     (boxplots, t-tests, chi-square, correlation)
      ↓
Feature Selection             (5 features selected)
      ↓
Machine Learning Modeling     (11 models, sklearn Pipeline, StandardScaler)
      ↓
SHAP Explainability           (beeswarm, bar, waterfall plots)
```

---

## 🤖 Models Compared

11 classifiers were evaluated using an sklearn `Pipeline` with `StandardScaler` preprocessing:

| Model | Test Accuracy | F1 (Alzheimer's) | CV Accuracy |
|---|---|---|---|
| **Random Forest** ⭐ | **96.74%** | **96.40%** | **96.51%** |
| Extra Trees | 95.81% | 95.50% | 95.60% |
| XGBoost | 94.65% | 94.20% | 94.40% |
| LightGBM | 94.42% | 93.90% | 94.10% |
| Gradient Boosting | 93.95% | 93.50% | 93.70% |
| SVM | 87.21% | 86.40% | 87.00% |
| MLP Neural Network | 88.60% | 87.90% | 88.20% |
| Logistic Regression | 85.12% | 84.70% | 85.00% |
| KNN | 82.33% | 81.50% | 82.10% |
| Decision Tree | 83.40% | 82.90% | 83.10% |
| Naive Bayes | 79.65% | 78.20% | 79.40% |

**Best model configuration:**
```python
RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=123)
```

---

## 💡 SHAP Explainability

SHAP (SHapley Additive exPlanations) was applied to interpret the Random Forest model's predictions.

Three plots were generated:

**1. Beeswarm Summary Plot** — shows distribution of SHAP values for all test samples.  
→ Low MMSE scores (blue) consistently produce large positive SHAP values, confirming cognitive impairment as the strongest predictor.

**2. Mean |SHAP| Bar Plot** — global feature importance ranking.  
→ MMSE ranks first by average impact, followed by ADL and FunctionalAssessment.

**3. Waterfall Plot** — individual prediction breakdown.  
→ Shows exactly which features pushed a single patient's prediction toward or away from an Alzheimer's diagnosis.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/mato8q/Alzheimer_s_Disease_PredictionNComprehensive_Data_Analysis_Using_ML.git
cd Alzheimer_s_Disease_PredictionNComprehensive_Data_Analysis_Using_ML
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap plotly
```

### 3. Add the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) and place at:
```
/kaggle/input/alzheimers-disease-dataset/alzheimers_disease_data.csv
```
Or update the path in the notebook's first cell.

### 4. Run the notebook
```bash
jupyter notebook alzheimer-s-disease-prediction-and-comprehensive-d.ipynb
```
Run all cells in order. The SHAP section is at the end (Section 5).

---

## 📁 Project Structure

```
├── alzheimer-s-disease-prediction-and-comprehensive-d.ipynb   # Main notebook
├── index.html                                                  # Project showcase website
├── presentation_note_compressed (2).pdf                       # Presentation slides
├── shap_summary_beeswarm.png                                  # Generated SHAP plot
├── shap_bar_importance.png                                    # Generated SHAP plot
├── shap_waterfall_individual.png                              # Generated SHAP plot
└── README.md
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `scikit-learn` | Pipeline, models, evaluation |
| `xgboost` / `lightgbm` | Boosting models |
| `shap` | Model explainability |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Static visualization |
| `plotly` | Interactive visualization |
| `scipy` | Statistical tests (t-test, chi-square) |

---

## 📄 License

This project was created for academic purposes as a Final Project for the Big Data Analytics course at National Dong Hwa University.

---

<div align="center">
  <sub>Made by <a href="https://mato8q.github.io/imnotmato8q.github.io/">Supharada Jundok</a> · CSIE, NDHU · 2025</sub>
</div>
