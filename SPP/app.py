import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import io
import base64
from flask import Flask, render_template

# Load and preprocess dataset
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "health", "famrel", "failures", "absences"]]
predict = "G3"

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and target
X = data.drop([predict], axis=1)
y = data[predict]

# Train-test split with random_state for consistent results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Random Forest Model with random_state
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test) * 100  # Convert to percentage
rf_predictions = rf_model.predict(X_test)

# Random Forest Metrics
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
rf_cv_scores = [round(score * 100, 2) for score in cross_val_score(rf_model, X, y, cv=5, scoring='r2', n_jobs=-1)]  # Convert to percentage

# Save Random Forest Model
with open("studentgrades_random_forest.pickle", "wb") as f:
    pickle.dump(rf_model, f)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_accuracy = lr_model.score(X_test, y_test) * 100  # Convert to percentage
lr_predictions = lr_model.predict(X_test)

# Linear Regression Metrics
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
lr_cv_scores = [round(score * 100, 2) for score in cross_val_score(lr_model, X, y, cv=5, scoring='r2', n_jobs=-1)]  # Convert to percentage

# Flask Application
app = Flask(__name__)

@app.route('/')
def home():
    # Generate Random Forest Plot
    rf_plot_url = create_plot(y_test, rf_predictions, title="Random Forest: Actual vs Predicted Grades")

    # Generate Linear Regression Plot
    lr_plot_url = create_plot(y_test, lr_predictions, title="Linear Regression: Actual vs Predicted Grades")

    return render_template(
        'index.html', 
        rf_accuracy=f"{rf_accuracy:.2f}%", lr_accuracy=f"{lr_accuracy:.2f}%",  # Format percentages
        rf_mae=rf_mae, rf_mse=rf_mse, rf_r2=rf_r2, rf_cv_scores=rf_cv_scores,
        lr_mae=lr_mae, lr_mse=lr_mse, lr_r2=lr_r2, lr_cv_scores=lr_cv_scores,
        rf_plot_url=rf_plot_url, lr_plot_url=lr_plot_url
    )

def create_plot(y_actual, y_predicted, title):
    # Create scatter plot for actual vs predicted grades
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_actual, y_predicted, alpha=0.7, color="blue")
    ax.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', linestyle="--")
    ax.set_xlabel("Actual Grades")
    ax.set_ylabel("Predicted Grades")
    ax.set_title(title)

    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return plot_base64

if __name__ == "__main__":
    app.run(debug=True)
