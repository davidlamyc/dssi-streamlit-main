import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

# Generate synthetic height data (in cm)
np.random.seed(0)
heights = np.random.normal(loc=170, scale=10, size=50)  # Mean 170 cm, std 10

# Generate corresponding weights (in kg) with some noise: weight ≈ 0.9 * height - 60
weights = 0.9 * heights - 60 + np.random.normal(loc=0, scale=5, size=50)

# Create a DataFrame
df = pd.DataFrame({'Height_cm': heights, 'Weight_kg': weights})

# Prepare features and target
X = df[['Height_cm']]
y = df['Weight_kg']

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the model
filename = 'linear_regression_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# # Predict weights
# df['Predicted_Weight'] = model.predict(X)

# # Show model coefficients
# print(f"Intercept: {model.intercept_:.2f}")
# print(f"Slope (weight per cm): {model.coef_[0]:.2f}")

# # Plot
# plt.scatter(df['Height_cm'], df['Weight_kg'], label='Actual data')
# plt.plot(df['Height_cm'], df['Predicted_Weight'], color='red', label='Regression line')
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.title('Linear Regression: Predicting Weight from Height')
# plt.legend()
# plt.show()

# # Display the first few rows of the dataset
# print(df.head())