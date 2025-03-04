import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Load the dataset
data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

# Select the value we want to predict
predict = "G3"

# List the variables to use for our predictions
data = data[["G1", "G2", "G3", "studytime", "health", "famrel", "failures", "absences"]]

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Define features (X) and target (y)
X = data.drop([predict], axis=1)
y = data[predict]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy: ", accuracy)

# Save the model
with open("studentgrades_random_forest.pickle", "wb") as f:
    pickle.dump(model, f)

# Load the model
with open("studentgrades_random_forest.pickle", "rb") as f:
    loaded_model = pickle.load(f)

# Make predictions
predictions = loaded_model.predict(X_test)

# Calculate and print average predicted and actual grades
average_predicted = np.mean(predictions)
average_actual = np.mean(y_test)
print("Average Predicted Final Grade:", average_predicted)
print("Average Actual Final Grade:", average_actual)

# Create visualization of predictions vs actual grades
plt.scatter(y_test, predictions)
plt.xlabel("Actual Final Grade")
plt.ylabel("Predicted Final Grade")
plt.title("Actual vs Predicted Final Grades")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 45-degree line
plt.show()
