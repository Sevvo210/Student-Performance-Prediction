# Introduction
This report outlines the development of a machine learning project aimed at predicting student performance using the "Student Performance Prediction" dataset. This dataset was taken from Kaggle. The project applies two regression algorithms—Random Forest and Linear Regression—to estimate students' final grades (G3). The dataset contains features such as study time, health, family relations, failures, and absences, which are analyzed to predict the target variable, G3.
The report presents the dataset preprocessing steps, the methodology used to build the models, performance metrics, and a comparison of the two algorithms.
# Methodology
Preprocessing:
1.	The dataset was shuffled to randomize the rows and improve model generalization.
2.	Selected 7 independent variables (features) and one dependent variable (G3).
3.	Split the dataset into training (90%) and testing (10%) sets.
   
Models:
1.	Random Forest Regressor:
o	Ensemble method using 100 decision trees.
o	Captures non-linear relationships.
o	Evaluated using cross-validation (5 folds).
2.	Linear Regression:
o	Assumes linear relationships between features and target.
o	Evaluated using cross-validation (5 folds).

# Results and Analysis
Visual Comparison and Outputs of the Code:

Scatter plots were generated to compare the actual vs. predicted grades for both models. The Random Forest model exhibited a closer fit to the actual values compared to Linear Regression, as seen in the alignment of points along the diagonal line.
# Screenshots of the Output
![Image](https://github.com/user-attachments/assets/c9a626ed-cfe3-4457-875d-33100ccbef88)

![Image](https://github.com/user-attachments/assets/f8b201d2-2989-467d-bd81-276c311f36c3)

# Conclusion
This project successfully implemented machine learning models to predict student performance. The Random Forest algorithm proved to be more effective, providing higher accuracy and lower error rates. These predictions could help educators identify students at risk of underperformance and provide targeted support.



