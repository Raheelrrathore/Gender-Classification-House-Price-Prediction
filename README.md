# Gender-Classification-House-Price-Prediction
📁 Overview
This repository contains two distinct machine learning projects developed by Raheel Riaz Rathore. Each project demonstrates the application of classical machine learning algorithms for solving real-world problems — one in the domain of classification and the other in regression.

📌 Project 1: Gender Classification
🎯 Objective
To predict the gender (Male/Female) of an individual based on specific facial features using supervised classification algorithms.

📊 Dataset
File: gender identification101.csv

Source: Local file path used in Google Colab

Target Variable: gender

Features Used:

long_hair

forehead_width_cm

forehead_height_cm

nose_wide

nose_long

lips_thin

distance_nose_to_lip_long

🔧 Preprocessing Steps
Removed missing and duplicate rows.

Label-encoded the categorical features (Male → 1, Female → 0).

Verified column data types and ranges.

Performed encoding for multiple facial attributes.

📈 Data Visualization
Pair plots and scatter plots to analyze gender distribution.

Correlation heatmap for feature relationships.

Bar and count plots for categorical feature exploration.

Kernel density estimation (KDE) for class distribution.

🧠 Models Implemented
Model	Description
Logistic Regression	Linear classification model for binary output
K-Nearest Neighbors (KNN)	Instance-based learning with distance-based voting
Linear Support Vector Classifier (SVC)	Finds the optimal hyperplane for classification
Random Forest Classifier	Ensemble method with decision trees
Bernoulli Naive Bayes	Probabilistic model assuming binary features

📊 Evaluation Metrics
Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

ROC Curve (plotted for Logistic Regression)

📌 Project 2: House Price Prediction
🎯 Objective
To predict house prices based on features such as size, number of rooms, and location using regression algorithms.

📊 Dataset
File: LRHOUSE.csv

Target Variable: price

Key Features:

bedrooms

bathrooms

floors

sqft_living, sqft_lot, etc.

location attributes (street, city, statezip, country)

🔧 Preprocessing Steps
Dropped non-numeric columns like date, and later street, city, statezip, country after encoding.

Converted categorical columns using Label Encoding.

Converted data types (e.g., bedrooms, bathrooms) to integers.

Handled null values and removed duplicate rows.

📈 Data Visualization
Correlation heatmap to examine relationships between features and price.

Actual vs Predicted scatter plot to visually assess model performance.

🧠 Models Implemented
Model	Description
Linear Regression	Basic regression model for numerical prediction
Random Forest Regressor	Ensemble-based model that averages multiple trees
Gradient Boosting Regressor	Boosted ensemble of weak learners for higher accuracy
Linear SVR (Support Vector Regression)	Regression variant of SVMs for continuous targets

📊 Evaluation Metrics
R² Score

Mean Absolute Error (MAE)

Train-Test Split: 70% training / 30% testing

🛠 Technologies Used
Languages & Libraries:

Python, NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn (sklearn)

Tools:

Google Colab (Notebook Environment)

CSV files for datasets

📁 Directory Structure
bash
Copy
Edit
├── gender_classification.py          # Main Python script containing both projects
├── gender identification101.csv     # Dataset for gender classification
├── LRHOUSE.csv                       # Dataset for house price prediction
└── README.md                         # This file
📌 How to Run
Upload the required CSV files to your environment.

Run the script using a Python IDE or Jupyter Notebook.

Make sure all required libraries (sklearn, matplotlib, seaborn, etc.) are installed.

🙋‍♂️ Author
Raheel Riaz Rathore

