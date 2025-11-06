"""
Lab 6: Heart Disease Prediction using Logistic Regression
Binary Classification with Comprehensive Preprocessing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
import os
warnings.filterwarnings('ignore')

# Load data
print("=" * 60)
print("LAB 6: HEART DISEASE PREDICTION")
print("=" * 60)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "heart_disease.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at: {csv_path}\nPlease ensure heart_disease.csv is in the same directory as this script.")

df = pd.read_csv(csv_path)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check missing values
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0].sort_values(ascending=False))

# Prepare target
target_col = "Heart Disease Status"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found")

y_raw = df[target_col].copy()
if y_raw.dtype == 'object':
    y = y_raw.map(lambda x: 1 if str(x).lower() == 'yes' else 0).astype(int)
else:
    y = y_raw.values

print(f"\nTarget distribution:")
print(pd.Series(y).value_counts())
print(f"\nClass balance:")
print(pd.Series(y).value_counts(normalize=True))

# Prepare features
X = df.drop(columns=[target_col]).copy()

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Drop constant columns
nunique = X.nunique()
constant_cols = nunique[nunique <= 1].index.tolist()
if constant_cols:
    print(f"\nDropping constant columns: {constant_cols}")
    X = X.drop(columns=constant_cols)
    numeric_cols = [c for c in numeric_cols if c not in constant_cols]
    categorical_cols = [c for c in categorical_cols if c not in constant_cols]

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Model pipeline
logistic_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', logistic_model)
])

# Hyperparameter tuning
print("\n" + "-" * 60)
print("HYPERPARAMETER TUNING")
print("-" * 60)

# Use balanced class weights by default for imbalanced data
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__class_weight': ['balanced']  # Focus on balanced weights for imbalanced data
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")

# Predictions
best_model = grid_search.best_estimator_
y_proba = best_model.predict_proba(X_test)[:, 1]

# Adjust threshold for imbalanced data (default 0.5, but can optimize)
# Using 0.5 threshold
y_pred = best_model.predict(X_test)

# Check prediction distribution
print(f"\nPrediction distribution:")
print(f"Class 0 (No Disease): {np.sum(y_pred == 0)}")
print(f"Class 1 (Disease): {np.sum(y_pred == 1)}")

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
print(f"              No Disease  Disease")
print(f"Actual No         {cm[0,0]:4d}      {cm[0,1]:4d}")
print(f"       Yes        {cm[1,0]:4d}      {cm[1,1]:4d}")

# Feature importance
print("\n" + "-" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("-" * 60)

# Get feature names after preprocessing
preprocessor.fit(X_train)
feature_names = numeric_cols.copy()

if categorical_cols:
    try:
        onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = list(onehot.get_feature_names_out(categorical_cols))
        feature_names.extend(cat_features)
    except:
        print("Could not extract categorical feature names")

coefs = best_model.named_steps['classifier'].coef_[0]
if len(coefs) == len(feature_names):
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    })
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    print("\nTop 15 features by importance:")
    print(feature_importance.head(15).to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Confusion Matrix
im = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Reds')
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_xticklabels(['No Disease', 'Disease'])
axes[0, 0].set_yticklabels(['No Disease', 'Disease'])
for i in range(2):
    for j in range(2):
        axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center', 
                        color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
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

# Feature Importance (if available)
if len(coefs) == len(feature_names):
    top_features = feature_importance.head(15)
    axes[1, 0].barh(range(len(top_features)), top_features['Coefficient'], color='crimson')
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels(top_features['Feature'], fontsize=8)
    axes[1, 0].set_xlabel('Coefficient Value')
    axes[1, 0].set_title('Top 15 Feature Coefficients')
    axes[1, 0].grid(axis='x', alpha=0.3)

# Prediction Probability Distribution
axes[1, 1].hist(y_proba[y_test == 0], bins=30, alpha=0.6, label='No Disease', color='green')
axes[1, 1].hist(y_proba[y_test == 1], bins=30, alpha=0.6, label='Disease', color='red')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Probability Distribution')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
output_path = os.path.join(script_dir, 'lab6_results.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved as '{output_path}'")
try:
    plt.show()
except:
    print("Note: Display not available. Plot saved to file.")

# Save results summary
results_summary = pd.DataFrame([{
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1_Score': f1,
    'ROC_AUC': roc_auc,
    'Best_C': grid_search.best_params_['classifier__C'],
    'Best_Class_Weight': grid_search.best_params_['classifier__class_weight']
}])
summary_path = os.path.join(script_dir, 'lab6_results_summary.csv')
results_summary.to_csv(summary_path, index=False)
print(f"\nResults summary saved as '{summary_path}'")

print("\n" + "=" * 60)
print("LAB 6 COMPLETED")
print("=" * 60)

