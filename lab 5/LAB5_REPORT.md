# Lab 5 Report: Pulsar Star Classification using Logistic Regression

## Introduction

Pulsar stars are highly magnetized, rotating neutron stars that emit beams of electromagnetic radiation. Identifying pulsars from radio telescope data is a challenging binary classification problem. This lab implements a Logistic Regression classifier to distinguish pulsar stars from non-pulsar signals.

## Objectives

1. Implement Logistic Regression for binary classification
2. Perform comprehensive data preprocessing and feature scaling
3. Optimize hyperparameters using Grid Search with Cross-Validation
4. Evaluate model performance using multiple metrics
5. Analyze feature importance and model interpretability

## Dataset

The pulsar star dataset contains 8 features derived from radio telescope observations:

1. Mean of the integrated profile
2. Standard deviation of the integrated profile
3. Excess kurtosis of the integrated profile
4. Skewness of the integrated profile
5. Mean of the DM-SNR curve
6. Standard deviation of the DM-SNR curve
7. Excess kurtosis of the DM-SNR curve
8. Skewness of the DM-SNR curve

**Target Variable**: Binary classification (0 = Non-Pulsar, 1 = Pulsar)

## Methodology

### Data Preprocessing

1. **Data Loading**: Loaded CSV file and examined structure
2. **Target Preparation**: Ensured target column is properly named and binary
3. **Missing Value Handling**: Filled missing values with column means
4. **Feature Scaling**: Applied StandardScaler to normalize features
5. **Train-Test Split**: 80% training, 20% testing with stratification

### Logistic Regression

Logistic Regression models the probability of a binary outcome using the logistic function:

```
P(y=1|x) = 1 / (1 + e^(-z))
where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Key Characteristics:**
- Outputs probabilities between 0 and 1
- Uses sigmoid activation function
- Interpretable coefficients
- Works well for linearly separable data

### Hyperparameter Tuning

**Grid Search Parameters:**
- **C (Regularization strength)**: [0.01, 0.1, 1, 10, 100]
  - Smaller C = stronger regularization
  - Larger C = weaker regularization
- **class_weight**: [None, 'balanced']
  - None: Equal weight to all classes
  - 'balanced': Adjusts weights inversely proportional to class frequencies

**Cross-Validation:**
- 5-fold cross-validation
- Scoring metric: ROC-AUC (Area Under ROC Curve)
- Selects best parameters based on average CV score

## Code Implementation

### Main Implementation Steps

```python
# Load and prepare data
df = pd.read_csv("pulsar_stars (1).csv")
X = df.drop(columns=['target'])
y = df['target']

# Handle missing values
X = X.fillna(X.mean())

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced']
}

logistic_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
grid_search = GridSearchCV(logistic_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
```

### Evaluation Metrics

1. **Accuracy**: Overall correctness of predictions
2. **Precision**: Proportion of positive predictions that are correct
3. **Recall**: Proportion of actual positives correctly identified
4. **F1-Score**: Harmonic mean of precision and recall
5. **ROC-AUC**: Area under ROC curve, measures classifier's ability to distinguish classes

## Results and Analysis

### Model Performance

The optimized Logistic Regression model achieves:
- High accuracy in distinguishing pulsars from non-pulsars
- Balanced precision and recall
- Strong ROC-AUC score indicating good class separation

### Feature Importance

The model identifies which features are most predictive:
- Features with larger absolute coefficients contribute more to predictions
- Positive coefficients increase probability of pulsar class
- Negative coefficients decrease probability of pulsar class

### Confusion Matrix Analysis

The confusion matrix shows:
- **True Negatives (TN)**: Correctly identified non-pulsars
- **False Positives (FP)**: Non-pulsars incorrectly classified as pulsars
- **False Negatives (FN)**: Pulsars incorrectly classified as non-pulsars
- **True Positives (TP)**: Correctly identified pulsars

### ROC Curve Analysis

The ROC curve demonstrates:
- Model's ability to distinguish between classes at different thresholds
- Trade-off between True Positive Rate and False Positive Rate
- AUC value close to 1.0 indicates excellent classification performance

## Visualizations

1. **Confusion Matrix Heatmap**: Visual representation of classification results
2. **ROC Curve**: Shows model's discrimination ability
3. **Feature Importance**: Bar chart of top 10 most important features
4. **Prediction Probability Distribution**: Histogram showing probability distributions for each class

## Discussion

### Model Strengths

1. **Interpretability**: Coefficients provide clear feature importance
2. **Efficiency**: Fast training and prediction
3. **Probability Output**: Provides confidence scores for predictions
4. **Regularization**: Prevents overfitting through C parameter

### Model Limitations

1. **Linear Decision Boundary**: Assumes linear relationship between features and log-odds
2. **Feature Scaling Required**: Sensitive to feature scales
3. **Class Imbalance**: May need class_weight adjustment for imbalanced data

### Applications

Logistic Regression is suitable for:
- Binary classification problems
- When interpretability is important
- Linearly separable data
- Baseline model for comparison

## Conclusion

This lab successfully implemented Logistic Regression for pulsar star classification. The model demonstrates strong performance with proper preprocessing and hyperparameter tuning. The interpretable coefficients provide insights into which features are most important for identifying pulsars, making this model valuable for both prediction and understanding the underlying patterns in the data.

## Future Improvements

1. Try non-linear models (Random Forest, SVM with RBF kernel)
2. Feature engineering to create interaction terms
3. Ensemble methods for improved performance
4. Deep learning approaches for complex patterns

## References

- Scikit-learn Logistic Regression documentation
- Pulsar dataset: HTRU2 dataset
- Classification metrics and evaluation techniques

---

**Note**: Run `lab5_pulsar_classification.py` to reproduce all results and generate visualizations.

