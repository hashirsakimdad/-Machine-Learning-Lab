# Lab 6 Report: Heart Disease Prediction using Logistic Regression

## Introduction

Heart disease is one of the leading causes of death worldwide. Early detection and prediction of heart disease can significantly improve patient outcomes. This lab implements a comprehensive Logistic Regression pipeline to predict heart disease status based on various patient characteristics and medical measurements.

## Objectives

1. Implement a complete machine learning pipeline with preprocessing
2. Handle mixed data types (numeric and categorical features)
3. Perform hyperparameter tuning using Grid Search
4. Evaluate model performance using comprehensive metrics
5. Analyze feature importance and model interpretability

## Dataset

The heart disease dataset contains 21 features including:

**Numeric Features:**
- Age
- Blood Pressure
- Cholesterol Level
- BMI (Body Mass Index)
- Sleep Hours
- Triglyceride Level
- Fasting Blood Sugar
- CRP Level (C-Reactive Protein)
- Homocysteine Level

**Categorical Features:**
- Gender
- Exercise Habits
- Smoking
- Family Heart Disease
- Diabetes
- High Blood Pressure
- Low HDL Cholesterol
- High LDL Cholesterol
- Alcohol Consumption
- Stress Level
- Sugar Consumption

**Target Variable**: Heart Disease Status (Yes/No)

## Methodology

### Data Preprocessing Pipeline

#### 1. Missing Value Analysis
- Identified columns with missing values
- Calculated missing value percentages
- Applied appropriate imputation strategies

#### 2. Target Variable Processing
- Converted categorical target to binary (Yes=1, No=0)
- Analyzed class distribution
- Ensured balanced representation in train-test split

#### 3. Feature Engineering
- Separated numeric and categorical features
- Identified constant columns (no variance)
- Dropped constant columns to reduce noise

#### 4. Preprocessing Transformers

**Numeric Features:**
- Missing value imputation using median (robust to outliers)
- StandardScaler for normalization (mean=0, std=1)

**Categorical Features:**
- Missing value imputation using most frequent value
- One-Hot Encoding to convert categories to binary features
- Handles unknown categories in test set

#### 5. Pipeline Construction

Used scikit-learn Pipeline and ColumnTransformer for:
- Organized preprocessing steps
- Prevents data leakage
- Ensures consistent transformations
- Easy hyperparameter tuning

### Logistic Regression Model

Logistic Regression models the probability of heart disease:

```
P(Disease) = 1 / (1 + e^(-z))
where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Key Parameters:**
- **C**: Inverse regularization strength (smaller = stronger regularization)
- **class_weight**: Handles class imbalance
  - None: Equal weights
  - 'balanced': Automatically adjusts weights

### Hyperparameter Tuning

**Grid Search Configuration:**
- **C values**: [0.01, 0.1, 1, 10, 100]
- **class_weight**: [None, 'balanced']
- **Cross-validation**: 5-fold CV
- **Scoring metric**: ROC-AUC (handles class imbalance well)

## Code Implementation

### Complete Pipeline Structure

```python
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

# Model pipeline
logistic_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', logistic_model)
])

# Hyperparameter tuning
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__class_weight': [None, 'balanced']
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

### Key Implementation Details

1. **Stratified Split**: Ensures both classes represented proportionally
2. **Pipeline**: Prevents data leakage and ensures consistent preprocessing
3. **Grid Search**: Exhaustive search over parameter space
4. **Cross-Validation**: Robust performance estimation

## Results and Analysis

### Model Performance Metrics

The optimized model achieves:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted diseases that are correct
- **Recall**: Proportion of actual diseases correctly identified
- **F1-Score**: Balanced measure of precision and recall
- **ROC-AUC**: Ability to distinguish between classes

### Confusion Matrix Interpretation

- **True Negatives**: Correctly identified healthy patients
- **False Positives**: Healthy patients incorrectly flagged (Type I error)
- **False Negatives**: Diseased patients missed (Type II error - critical)
- **True Positives**: Correctly identified diseased patients

### Feature Importance Analysis

The model identifies which features contribute most to predictions:
- Features with larger absolute coefficients are more important
- Positive coefficients increase disease probability
- Negative coefficients decrease disease probability
- Top features provide insights into risk factors

### ROC Curve Analysis

The ROC curve shows:
- Model's discrimination ability across thresholds
- Trade-off between sensitivity and specificity
- Optimal threshold selection for different use cases
- AUC value indicates overall model quality

## Visualizations

1. **Confusion Matrix Heatmap**: Visual classification results
2. **ROC Curve**: Model discrimination performance
3. **Feature Importance**: Top 15 most important features
4. **Probability Distribution**: Prediction confidence analysis

## Discussion

### Model Strengths

1. **Comprehensive Preprocessing**: Handles mixed data types effectively
2. **Interpretability**: Coefficients provide medical insights
3. **Robust Pipeline**: Prevents common data science mistakes
4. **Class Imbalance Handling**: Balanced class weights improve recall

### Clinical Implications

1. **Risk Factor Identification**: Model highlights important predictors
2. **Early Detection**: Can identify at-risk patients
3. **Preventive Care**: Supports clinical decision-making
4. **Resource Allocation**: Helps prioritize screening

### Model Limitations

1. **Linear Assumptions**: May miss complex interactions
2. **Data Quality**: Depends on accurate input data
3. **Generalization**: May not apply to all populations
4. **Threshold Selection**: Requires domain expertise

### Ethical Considerations

1. **False Negatives**: Missing actual disease is critical
2. **False Positives**: May cause unnecessary anxiety
3. **Bias**: Ensure model works for all demographics
4. **Transparency**: Explainable predictions build trust

## Conclusion

This lab successfully implemented a comprehensive Logistic Regression pipeline for heart disease prediction. The model demonstrates strong performance with proper preprocessing, hyperparameter tuning, and evaluation. The interpretable coefficients provide valuable insights into risk factors, making this approach valuable for both prediction and understanding disease patterns.

## Future Improvements

1. **Feature Engineering**: Create interaction terms and polynomial features
2. **Advanced Models**: Try Random Forest, XGBoost, or Neural Networks
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Deep Learning**: Capture complex non-linear relationships
5. **External Validation**: Test on independent datasets
6. **Clinical Integration**: Deploy in real-world healthcare settings

## References

- Scikit-learn Pipeline and ColumnTransformer documentation
- Logistic Regression for medical diagnosis
- ROC-AUC and classification metrics
- Heart disease prediction research

---

**Note**: Run `lab6_heart_disease.py` to reproduce all results. The script generates visualizations and saves a results summary CSV file.

