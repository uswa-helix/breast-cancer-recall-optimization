import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    make_scorer,      # Added for custom scoring
    recall_score      # Added for custom scoring
)

# ==========================================
# 1. Data Loading & Preparation
# ==========================================
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names  # 0: 'malignant', 1: 'benign'

# Split the data (80% training, 20% testing)
# ADDED: stratify=y to ensure representative class distributions in both splits
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. Baseline Model & Feature Importance
# ==========================================
base_model = RandomForestClassifier(random_state=42)
base_model.fit(Xtr, ytr)

print("--- Top 5 Diagnostic Cytological Features ---")
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': base_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance_df.head(5).to_string(index=False))
print("\n")

# ==========================================
# 3. Diagnosing Baseline False Negatives
# ==========================================
print("--- Diagnosing Baseline Missed Malignant Cases ---")
y_probs_base = base_model.predict_proba(Xte)

for i in range(len(yte)):
    true_class = "Malignant" if yte[i] == 0 else "Benign"
    prob_malignant = y_probs_base[i, 0]

    # Identify cases where a malignant tumor was assigned a low probability
    if yte[i] == 0 and prob_malignant < 0.50:
        print(f"Sample {i}: True Label = {true_class} | Model assigned Malignant Prob: {prob_malignant:.2%}")
print("\n")

# ==========================================
# 4. Optimized Clinical Model (Pipeline & GridSearchCV)
# ==========================================
print("--- Tuning and Training Optimized Clinical Model ---")

# Define the pipeline with a scaler and the random forest classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Define the hyperparameter grid to search over
param_grid = {
    'rf__n_estimators': [50, 100, 200, 300],
    'rf__max_depth': [None, 5, 10, 20],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Use StratifiedKFold to maintain the class distribution in each fold
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ADDED: Create a custom scorer specifically for catching malignant (0) cases
malignant_recall = make_scorer(recall_score, pos_label=0)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring=malignant_recall,  # ADDED: Optimize specifically for malignant recall
    n_jobs=-1,                 # Use all available CPU cores
    verbose=1
)

# Fit the GridSearch to find the best parameters
grid_search.fit(Xtr, ytr)

print(f"\nBest Hyperparameters Found:")
for param, value in grid_search.best_params_.items():
    print(f" - {param}: {value}")
print("\n")

# Extract the best model from the grid search
final_model = grid_search.best_estimator_
final_preds = final_model.predict(Xte)

# ==========================================
# 5. Final Evaluation Metrics & Visualization
# ==========================================
print(f"Final Model Overall Accuracy: {final_model.score(Xte, yte):.4f}\n")
print("--- Final Classification Report (Default 50% Threshold) ---")
print(classification_report(yte, final_preds, target_names=target_names))

# Plot the Final Confusion Matrix
cm_final = confusion_matrix(yte, final_preds)
disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=target_names)
disp_final.plot(cmap=plt.cm.Purples)
plt.title("Optimized Diagnostic Confusion Matrix (50% Threshold)")
plt.show()

# ==========================================
# 6. Clinical Threshold Adjustment (30%) & ROC Visualization
# ==========================================
print("\n--- Applying Custom Clinical Threshold (30%) ---")

# Get the probability array for the testing set
final_probs = final_model.predict_proba(Xte)
# Extract just the probabilities for the 'malignant' class (Index 0)
malignant_probs = final_probs[:, 0]

# Create new predictions: 0 (malignant) if probability >= 30%, else 1 (benign)
custom_threshold = 0.30
clinical_preds = np.where(malignant_probs >= custom_threshold, 0, 1)

print("--- Clinical Classification Report (30% Threshold) ---")
print(classification_report(yte, clinical_preds, target_names=target_names))

# Plot the Threshold-Adjusted Confusion Matrix
cm_clinical = confusion_matrix(yte, clinical_preds)
disp_clinical = ConfusionMatrixDisplay(confusion_matrix=cm_clinical, display_labels=target_names)
disp_clinical.plot(cmap=plt.cm.Reds)
plt.title("Clinical Confusion Matrix (30% Threshold)")
plt.show()

# Plot the ROC Curve
print("\n--- Generating ROC Curve ---")
# Setting pos_label=0 because 'malignant' is encoded as 0 in this dataset
roc_display = RocCurveDisplay.from_predictions(
    yte,
    malignant_probs,
    pos_label=0,
    name="Optimized RF Pipeline",
    color="darkorange"
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--") # Add a baseline random-chance diagonal
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.show()


import joblib

# Save the pipeline to a file
joblib.dump(grid_search.best_estimator_, 'clinical_rf_pipeline.pkl')
print("Model successfully saved!")
