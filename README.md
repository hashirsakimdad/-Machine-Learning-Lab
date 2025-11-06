# Machine Learning Lab Repository

Complete collection of Machine Learning laboratory assignments and implementations covering regression, classification, and data preprocessing techniques.

## 📚 Repository Overview

This repository contains professional implementations of Machine Learning algorithms and techniques, including comprehensive reports, clean code, and detailed documentation for each lab assignment.

## 📁 Directory Structure

```
ML LAB/
│
├── lab 1/                    # Lab 1 implementations
│   └── lab1.py
│
├── lab 2/                     # Lab 2 implementations
│   └── LAB 2/
│       └── task2.py
│
├── lab 3/                     # Lab 3 implementations
│   └── lab3.py
│
├── lab 4/                     # Regression Techniques
│   ├── lab4_task1.py          # Linear & Ridge Regression
│   ├── lab4_task2.py          # Lasso Regression
│   ├── LAB4_REPORT.md        # Comprehensive Report
│   ├── housing_data.csv       # Dataset
│   └── *.png                  # Result visualizations
│
├── lab 5/                     # Pulsar Star Classification
│   ├── lab5_pulsar_classification.py
│   ├── LAB5_REPORT.md
│   ├── pulsar_stars (1).csv
│   └── lab5_results.png
│
├── lab 6/                     # Heart Disease Prediction
│   ├── lab6_heart_disease.py
│   ├── LAB6_REPORT.md
│   ├── heart_disease.csv
│   ├── lab6_results.png
│   └── lab6_results_summary.csv
│
├── open handed lab/           # Open-ended Lab Project
│   ├── Section A/            # Data Preprocessing
│   ├── Section b/            # Exploratory Data Analysis
│   ├── Section C/            # Feature Engineering
│   ├── Section D/            # Model Training
│   └── Section E/            # Model Evaluation
│
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── IMPLEMENTATION_SUMMARY.md  # Implementation details
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Required packages (install using requirements.txt)

### Installation

```bash
# Clone or download this repository
# Navigate to the ML LAB directory

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

## 📖 Lab Descriptions

### Lab 1: Introduction to Machine Learning
**File**: `lab1.py`

Basic ML concepts and data handling introduction.

### Lab 2: Data Analysis and Visualization
**File**: `lab 2/LAB 2/task2.py`

Data analysis techniques and visualization methods.

### Lab 3: Intermediate ML Techniques
**File**: `lab 3/lab3.py`

Intermediate machine learning concepts and implementations.

### Lab 4: Regression Techniques ⭐

**Objective**: Implement and compare Linear Regression, Ridge Regression, and Lasso Regression for housing price prediction.

#### Task 1: Linear and Ridge Regression
**File**: `lab 4/lab4_task1.py`

- **Techniques**: Linear Regression, Ridge Regression with L2 regularization
- **Dataset**: Housing data with price prediction
- **Features**: 
  - Data preprocessing and cleaning
  - Feature scaling
  - Model comparison (MSE, R², MAE)
  - Coefficient visualization
- **Run**:
  ```bash
  cd "lab 4"
  python lab4_task1.py
  ```

#### Task 2: Lasso Regression
**File**: `lab 4/lab4_task2.py`

- **Techniques**: Lasso Regression with L1 regularization, Cross-Validation
- **Features**:
  - Automatic feature selection
  - Optimal alpha selection using LassoCV
  - Feature importance analysis
- **Run**:
  ```bash
  cd "lab 4"
  python lab4_task2.py
  ```

**Report**: See `lab 4/LAB4_REPORT.md` for detailed methodology and analysis.

---

### Lab 5: Pulsar Star Classification ⭐

**Objective**: Binary classification of pulsar stars using Logistic Regression.

**File**: `lab 5/lab5_pulsar_classification.py`

- **Technique**: Logistic Regression with hyperparameter tuning
- **Dataset**: Pulsar star detection dataset (HTRU2)
- **Features**:
  - 8 statistical features from radio telescope observations
  - Grid Search with Cross-Validation
  - Comprehensive evaluation metrics
  - ROC curve and confusion matrix visualization
  - Feature importance analysis

**Run**:
```bash
cd "lab 5"
python lab5_pulsar_classification.py
```

**Outputs**:
- Model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrix visualization
- ROC curve
- Feature importance ranking
- Probability distribution analysis

**Report**: See `lab 5/LAB5_REPORT.md` for comprehensive analysis.

---

### Lab 6: Heart Disease Prediction ⭐

**Objective**: Predict heart disease using Logistic Regression with mixed data types.

**File**: `lab 6/lab6_heart_disease.py`

- **Technique**: Logistic Regression with comprehensive preprocessing pipeline
- **Dataset**: Heart disease dataset with 21 features
- **Features**:
  - Mixed data types (numeric + categorical)
  - Complete preprocessing pipeline (Pipeline + ColumnTransformer)
  - One-Hot Encoding for categorical features
  - Missing value imputation (median for numeric, most frequent for categorical)
  - Class imbalance handling
  - Hyperparameter tuning
  - Results export to CSV

**Run**:
```bash
cd "lab 6"
python lab6_heart_disease.py
```

**Outputs**:
- Model performance metrics
- Confusion matrix
- ROC curve
- Feature importance (top 15 features)
- Results summary CSV file

**Report**: See `lab 6/LAB6_REPORT.md` for detailed methodology and clinical insights.

---

### Open Handed Lab: Comprehensive ML Project

**Location**: `open handed lab/`

Complete machine learning project with multiple sections:
- **Section A**: Data Preprocessing (missing value handling, imputation)
- **Section b**: Exploratory Data Analysis (visualizations, statistics)
- **Section C**: Feature Engineering
- **Section D**: Model Training
- **Section E**: Model Evaluation (ROC curves, metrics)

## 🎯 Key Features

### Code Quality
- ✅ Clean, professional code structure
- ✅ Comprehensive error handling
- ✅ Detailed comments and documentation
- ✅ Reproducible results (fixed random seeds)
- ✅ Path-independent execution (works from any directory)

### Preprocessing
- ✅ Missing value handling
- ✅ Feature scaling and normalization
- ✅ Categorical encoding (One-Hot Encoding)
- ✅ Data cleaning and validation
- ✅ Feature selection

### Model Evaluation
- ✅ Multiple evaluation metrics
- ✅ Cross-validation
- ✅ Hyperparameter tuning (Grid Search)
- ✅ Professional visualizations
- ✅ Results export

### Documentation
- ✅ Comprehensive lab reports
- ✅ Code comments and explanations
- ✅ Methodology documentation
- ✅ Results analysis

## 📊 Lab Results Summary

### Lab 4: Regression
- **Linear Regression**: R² = 0.9098
- **Ridge Regression**: R² = 0.9098 (with regularization)
- **Lasso Regression**: Automatic feature selection with optimal alpha

### Lab 5: Pulsar Classification
- **Accuracy**: 96.96%
- **ROC-AUC**: 0.9727
- **Precision**: 78.44%
- **Recall**: 92.07%
- **F1-Score**: 84.71%

### Lab 6: Heart Disease Prediction
- **Complete preprocessing pipeline**
- **Mixed data type handling**
- **Class imbalance management**
- **Feature importance analysis**

## 🔧 Troubleshooting

### Common Issues

**Issue**: FileNotFoundError - Dataset not found
- **Solution**: Scripts automatically detect dataset location. Ensure CSV files are in the same directory as the Python scripts.

**Issue**: Import errors
- **Solution**: Install all requirements: `pip install -r requirements.txt`

**Issue**: Display errors (matplotlib)
- **Solution**: Plots are automatically saved even if display is unavailable. Check the script directory for PNG files.

**Issue**: Memory errors
- **Solution**: For large datasets, consider reducing features or using sparse matrices.

**Issue**: Convergence warnings
- **Solution**: Increase `max_iter` parameter or adjust regularization strength.

## 📝 Usage Examples

### Running Individual Labs

```bash
# Lab 4 - Task 1
cd "lab 4"
python lab4_task1.py

# Lab 4 - Task 2
python lab4_task2.py

# Lab 5
cd "../lab 5"
python lab5_pulsar_classification.py

# Lab 6
cd "../lab 6"
python lab6_heart_disease.py
```

### Viewing Reports

All labs include comprehensive markdown reports:
- `lab 4/LAB4_REPORT.md`
- `lab 5/LAB5_REPORT.md`
- `lab 6/LAB6_REPORT.md`

## 📈 Visualizations

Each lab generates professional visualizations:
- **Lab 4**: Feature coefficient comparisons, prediction scatter plots
- **Lab 5**: Confusion matrix, ROC curve, feature importance, probability distributions
- **Lab 6**: Confusion matrix, ROC curve, feature importance, probability distributions

All visualizations are saved as high-resolution PNG files (300 DPI).

## 🎓 Learning Outcomes

After completing these labs, you will understand:

1. **Regression Techniques**:
   - Linear Regression fundamentals
   - Regularization (L1 and L2)
   - Feature selection methods
   - Model comparison and evaluation

2. **Classification Techniques**:
   - Logistic Regression
   - Binary classification
   - Evaluation metrics
   - Hyperparameter tuning

3. **Data Preprocessing**:
   - Handling missing values
   - Feature scaling
   - Categorical encoding
   - Mixed data type processing

4. **Model Evaluation**:
   - Multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
   - Cross-validation
   - Visualization techniques

## 📚 Additional Resources

- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Pandas Documentation**: https://pandas.pydata.org/
- **Matplotlib Documentation**: https://matplotlib.org/
- **NumPy Documentation**: https://numpy.org/

## 👤 Author

Machine Learning Lab Assignments
- Semester 5 - Machine Learning Laboratory
- All implementations include professional code and comprehensive reports

## 📄 License

This work is for educational purposes as part of Machine Learning Lab coursework.

## 🔄 Version Information

- **Python**: 3.7+
- **Scikit-learn**: 1.0.0+
- **Pandas**: 1.3.0+
- **NumPy**: 1.21.0+
- **Matplotlib**: 3.4.0+

## ⚠️ Important Notes

1. All scripts use `random_state=42` for reproducibility
2. Scripts automatically detect their directory location
3. All outputs are saved in the script's directory
4. Datasets must be in the same directory as the Python scripts
5. Results may vary slightly with different random seeds

## 🎉 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Navigate to Lab Directory**:
   ```bash
   cd "lab 4"  # or lab 5, lab 6
   ```

3. **Run the Script**:
   ```bash
   python lab4_task1.py
   ```

4. **View Results**:
   - Check console output for metrics
   - View generated PNG files for visualizations
   - Read markdown reports for detailed analysis

---

**Happy Learning! 🚀**

For questions or issues, refer to the individual lab reports or check the code comments for detailed explanations.
