
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset from local CSV instead of sklearn's non-existent `housing_data`
# Expecting file `housing_data.csv` to be in the same folder as this script.
df = pd.read_csv("housing_data.csv", index_col=0)
print("Columns in CSV:", list(df.columns))

# Basic cleaning: strip column names, convert common Yes/No to binary
df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
df = df.replace({"Yes": 1, "No": 0, "yes": 1, "no": 0})

# Choose target column (price) - prefer 'new_price'
if 'new_price' in df.columns:
	y = df['new_price']
else:
	# fallback to first numeric column
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	if len(numeric_cols) == 0:
		raise ValueError("No numeric columns found to use as target in housing_data.csv")
	y = df[numeric_cols[0]]

# Select numeric features and drop the target column if present
X = df.select_dtypes(include=[np.number]).copy()
if y.name in X.columns:
	X = X.drop(columns=[y.name])

if X.shape[1] == 0:
	# If no numeric features, try to extract some simple features from text columns
	raise ValueError("No numeric feature columns found in CSV. Consider preprocessing categorical/text columns.")

print(X.head())

# Fill missing numeric values with column mean
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_model = Ridge(alpha=1.0)  # alpha = regularization strength
ridge_model.fit(X_train_scaled, y_train)

y_pred_ridge = ridge_model.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(" Linear Regression:")
print("MSE:", mse_linear)
print("R2 Score:", r2_linear)

print("\n Ridge Regression (with scaling):")
print("MSE:", mse_ridge)
print("R2 Score:", r2_ridge)

plt.figure(figsize=(10,6))
plt.barh(X.columns, linear_model.coef_, color='skyblue', label='Linear')
plt.barh(X.columns, ridge_model.coef_, color='orange', alpha=0.6, label='Ridge')
plt.xlabel("Coefficient Value")
plt.title("Feature Importance: Linear vs Ridge Regression")
plt.legend()
plt.show()
