# Step 1: Load and Explore the Insurance Dataset
# ------------------------------------------------
# This step loads the dataset, displays basic statistics, checks for missing values, and visualizes key features.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('insurance[1] (1).csv')

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Display info and summary statistics
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize distributions of key features

# Professional visualization with axis labels
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['age'], kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.histplot(df['charges'], kde=True, color='salmon')
plt.title('Charges Distribution')
plt.xlabel('Charges')
plt.ylabel('Count')


plt.figtext(1.00, 0.96, 'Made by Hashir Sakimdad', ha='right', va='top', fontsize=10, color='black')
plt.tight_layout()
plt.show()
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv("insurance[1] (1).csv")

print("First 5 Rows of Dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Data Preprocessing
# Convert categorical columns into numeric (One-Hot Encoding)
df_encoded = pd.get_dummies(df, drop_first=True)

print("\nAfter Encoding:")
print(df_encoded.head())

# Step 4: Define Features and Target
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2 Score):", r2)

# Step 9: Visualization (Actual vs Predicted)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.figtext(0.95, 0.96, 'Made by Hashir Sakimdad', ha='right', va='top', fontsize=10, color='gray')
plt.tight_layout()
plt.show()


# Step 10: Correlation Heatmap (Refined)

plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Heatmap", fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.figtext(0.95, 0.98, 'Made by Hashir Sakimdad', ha='right', va='top', fontsize=12, color='gray')
plt.tight_layout()
plt.show()
