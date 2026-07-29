#  Clinical Breast Cancer Detection: Recall Optimization

An end-to-end Machine Learning pipeline using `scikit-learn` designed to classify breast tissue biopsies as **Malignant** or **Benign**. This project focuses on **minimizing False Negatives** in medical diagnostics through custom threshold tuning and targeted cross-validation scoring.

---

##  Key Engineering & Clinical Features

* **Data Preprocessing & Scaling:** Implemented a Scikit-Learn `Pipeline` with `StandardScaler` to prevent data leakage during cross-validation.
* **Stratified Validation:** Used `StratifiedKFold` and stratified train/test splits to preserve class ratios across training and evaluation sets.
* **Custom Clinical Metric:** Utilized `make_scorer` with `recall_score` (targeting Class 0 / Malignant) during `GridSearchCV` to prioritize catching cancers over standard overall accuracy.
* **Threshold Optimization:** Lowered the decision threshold from 50% to 30% to achieve **100% malignant recall** on the test set.

---

##  Results & Visualizations

| Standard Model (50% Threshold) | Clinical Model (30% Threshold) |
| :---: | :---: |
| ![Standard Matrix](output%20b.PNG) | ![Clinical Matrix](output%20c.PNG) |
| **3 Missed Cancers (False Negatives)** | **0 Missed Cancers (100% Recall)** |

### ROC Performance Curve
![ROC Curve](output%20d.PNG)

---

##  Performance Summary

| Metric | Standard Model (50% Threshold) | Optimized Clinical Model (30% Threshold) |
| :--- | :--- | :--- |
| **Malignant Recall** | ~93% (3 Missed Cancers) | **100% (0 Missed Cancers)** |
| **ROC AUC Score** | 0.99 | 0.99 |

> **Clinical Trade-off:** By lowering the decision threshold to 30%, the model eliminated all False Negatives (missed cancer cases) at the cost of a small increase in False Positives (preventative false alarms), which aligns with real-world medical risk tolerance.

---

##  How to Run

1. Clone or download this repository.
2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn

Run the main script:

Bash

python breast_cancer_detection.py
