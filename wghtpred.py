# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from google.colab import files
from sklearn.preprocessing import OneHotEncoder

# Step 2: Upload the CSV file
uploaded = files.upload()

# Step 3: Read the CSV file
# Replace 'data.csv' with the actual file name you upload
df = pd.read_csv(list(uploaded.keys())[0])

# Step 4: Check the imported data (optional, just to see the first few rows)
print(df.head())

# Step 5: One-Hot Encode the categorical columns (Diet and Exercise)
df_encoded = pd.get_dummies(df, columns=['Diet', 'Exercise'], drop_first=True)

# Step 6: Split dataset into features (X) and target (y)
X = df_encoded.drop('Weight', axis=1)  # Features (all except the 'Weight' column)
y = df_encoded['Weight']  # Target (Weight)

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Predict weights using the test set
predictions = model.predict(X_test)

# Step 10: Output the predictions
print("Predicted Weights: ", predictions)

# Step 11: (Optional) Check the model's accuracy
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")
