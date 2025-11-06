import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load and preprocess data
df = pd.read_csv("housing_data.csv", index_col=0)
df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
df = df.replace({"Yes": 1, "No": 0, "yes": 1, "no": 0})

# Prepare features and target
y = df['new_price'] if 'new_price' in df.columns else df.select_dtypes(include=[np.number]).iloc[:, 0]
X = df.select_dtypes(include=[np.number]).copy()
if y.name in X.columns:
    X = X.drop(columns=[y.name])

X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso Regression with fixed alpha
lasso_model = Lasso(alpha=100.0, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

# Lasso with Cross-Validation for optimal alpha
lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 100), cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
y_pred_cv = lasso_cv.predict(X_test_scaled)

mse_cv = mean_squared_error(y_test, y_pred_cv)
r2_cv = r2_score(y_test, y_pred_cv)
mae_cv = mean_absolute_error(y_test, y_pred_cv)

# Display results
print("Lasso Regression (alpha=100.0):")
print(f"MSE: {mse_lasso:.2f}")
print(f"R2 Score: {r2_lasso:.4f}")
print(f"MAE: {mae_lasso:.2f}")

print("\nLasso with Cross-Validation:")
print(f"Optimal Alpha: {lasso_cv.alpha_:.2f}")
print(f"MSE: {mse_cv:.2f}")
print(f"R2 Score: {r2_cv:.4f}")
print(f"MAE: {mae_cv:.2f}")

# Feature selection analysis
n_features_used = np.sum(lasso_cv.coef_ != 0)
print(f"\nFeatures selected: {n_features_used}/{len(X.columns)}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Coefficient comparison
axes[0].barh(X.columns, lasso_model.coef_, alpha=0.7, label='Lasso (α=100)')
axes[0].barh(X.columns, lasso_cv.coef_, alpha=0.7, label='Lasso CV')
axes[0].set_xlabel("Coefficient Value")
axes[0].set_title("Feature Coefficients: Lasso vs Lasso CV")
axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)

# Plot 2: Prediction comparison
axes[1].scatter(y_test, y_pred_cv, alpha=0.6, edgecolors='k', linewidths=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel("Actual Price")
axes[1].set_ylabel("Predicted Price")
axes[1].set_title(f"Lasso CV Predictions (R² = {r2_cv:.4f})")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

