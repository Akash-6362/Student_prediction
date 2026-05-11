import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/StudentsPerformance.csv")

# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# Create Result Column
# Pass = 1 , Fail = 0
data['result'] = np.where(data['math score'] >= 40, 1, 0)

print("\nResult Column Added:")
print(data[['math score', 'result']].head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Encode categorical columns
encoder = LabelEncoder()

data['gender'] = encoder.fit_transform(data['gender'])
data['lunch'] = encoder.fit_transform(data['lunch'])

# Select Features
X = data[['gender', 'reading score', 'writing score']]

# Target Variable
y = data['result']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = DecisionTreeClassifier()

# Train Model
model.fit(X_train, y_train)

# Predict Results
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)


print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Graph 1 - Math Score Distribution

plt.figure(figsize=(8,5))

plt.hist(data['math score'], bins=10)

plt.title("Math Score Distribution")
plt.xlabel("Math Scores")
plt.ylabel("Number of Students")

plt.show()



# Graph 2 - Pass vs Fail Count

pass_fail = data['result'].value_counts()

plt.figure(figsize=(6,5))

plt.bar(['Pass', 'Fail'], pass_fail)

plt.title("Pass vs Fail Students")
plt.xlabel("Result")
plt.ylabel("Count")

plt.show()