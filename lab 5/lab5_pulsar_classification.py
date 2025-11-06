"""
Lab 5: Pulsar Star Classification
Binary Classification using Logistic Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Load data
print("=" * 60)
print("LAB 5: PULSAR STAR CLASSIFICATION")
print("=" * 60)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "pulsar_stars (1).csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at: {csv_path}\nPlease ensure pulsar_stars (1).csv is in the same directory as this script.")

df = pd.read_csv(csv_path)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Rename target column if needed
if 'target_class' in df.columns:
    df = df.rename(columns={'target_class': 'target'})
elif df.columns[-1] not in df.columns[:-1]:
    df = df.rename(columns={df.columns[-1]: 'target'})

# Prepare features and target
X = df.drop(columns=['target'])
y = df['target']

print(f"\nFeatures: {list(X.columns)}")
print(f"\nTarget distribution:")
print(y.value_counts())
print(f"\nClass balance: {y.value_counts(normalize=True)}")

# Handle missing values
X = X.fillna(X.mean())
print(f"\nMissing values after imputation: {X.isnull().sum().sum()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with Grid Search
print("\n" + "-" * 60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("-" * 60)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced']
}

logistic_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")

# Predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "-" * 60)
print("MODEL PERFORMANCE METRICS")
print("-" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\n" + "-" * 60)
print("CLASSIFICATION REPORT")
print("-" * 60)
print(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n" + "-" * 60)
print("CONFUSION MATRIX")
print("-" * 60)
print(f"                Predicted")
print(f"              Negative  Positive")
print(f"Actual Negative   {cm[0,0]:4d}     {cm[0,1]:4d}")
print(f"       Positive   {cm[1,0]:4d}     {cm[1,1]:4d}")

# Feature importance
print("\n" + "-" * 60)
print("FEATURE IMPORTANCE (Top 10)")
print("-" * 60)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_model.coef_[0],
    'Abs_Coefficient': np.abs(best_model.coef_[0])
})
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
print(feature_importance.head(10).to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Confusion Matrix Heatmap
im = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')
for i in range(2):
    for j in range(2):
        axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=14)
plt.colorbar(im, ax=axes[0, 0])

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0, 1].plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], 'r--', lw=2, label='Random Classifier')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Feature Importance
top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['Coefficient'], color='steelblue')
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['Feature'])
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_title('Top 10 Feature Coefficients')
axes[1, 0].grid(axis='x', alpha=0.3)

# Prediction Distribution
axes[1, 1].hist(y_proba[y_test == 0], bins=30, alpha=0.6, label='Non-Pulsar', color='blue')
axes[1, 1].hist(y_proba[y_test == 1], bins=30, alpha=0.6, label='Pulsar', color='red')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Probability Distribution')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
output_path = os.path.join(script_dir, 'lab5_results.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved as '{output_path}'")
try:
    plt.show()
except:
    print("Note: Display not available. Plot saved to file.")

print("\n" + "=" * 60)
print("LAB 5 COMPLETED")
print("=" * 60)

