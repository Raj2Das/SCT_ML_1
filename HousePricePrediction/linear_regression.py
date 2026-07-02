import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =====================================
# Create Graph Folder
# =====================================
os.makedirs("graphs", exist_ok=True)

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv("house_price.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

# =====================================
# Features and Target
# =====================================
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# =====================================
# Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# Train Model
# =====================================
model = LinearRegression()
model.fit(X_train, y_train)

# =====================================
# Prediction
# =====================================
y_pred = model.predict(X_test)

# =====================================
# Evaluation
# =====================================
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"R² Score : {r2:.4f}")
print(f"MSE      : {mse:,.2f}")

print("\n========== COEFFICIENTS ==========")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:<12}: {coef:.2f}")

print(f"Intercept   : {model.intercept_:.2f}")

# =====================================
# Example Prediction
# =====================================
new_house = pd.DataFrame({
    'area': [2000],
    'bedrooms': [3],
    'bathrooms': [2]
})

predicted_price = model.predict(new_house)

print("\n========== NEW HOUSE ==========")
print(f"Predicted Price: ₹ {predicted_price[0]:,.2f}")

# =====================================
# GRAPH 1
# =====================================
plt.figure(figsize=(8,6))
plt.scatter(df['area'], df['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs House Price")
plt.tight_layout()
plt.savefig("graphs/area_vs_price.png")
plt.show()

# =====================================
# GRAPH 2
# =====================================
plt.figure(figsize=(8,6))
plt.scatter(df['bedrooms'], df['price'])
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Bedrooms vs House Price")
plt.tight_layout()
plt.savefig("graphs/bedrooms_vs_price.png")
plt.show()

# =====================================
# GRAPH 3
# =====================================
plt.figure(figsize=(8,6))
plt.scatter(df['bathrooms'], df['price'])
plt.xlabel("Bathrooms")
plt.ylabel("Price")
plt.title("Bathrooms vs House Price")
plt.tight_layout()
plt.savefig("graphs/bathrooms_vs_price.png")
plt.show()

# =====================================
# GRAPH 4
# =====================================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()

plt.savefig("graphs/actual_vs_predicted.png")
plt.show()

print("\n✅ All graphs saved successfully!")
print("📁 Location: HousePricePrediction/graphs/")