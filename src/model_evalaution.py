import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dvclive import Live

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=35)
n_estimators=100
max_depth=10
# Train the RandomForest model
model = RandomForestClassifier(random_state=42, max_depth=max_depth, n_estimators=n_estimators)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
with Live(save_dvc_exp=True) as live:
# Print the results
    live.log_metric(f'Accuracy: {accuracy:.2f}')
    live.log_metricint(f'F1 Score: {f1:.2f}')
    live.log_metric(f'Precision: {precision:.2f}')
    live.log_metric(f'Recall: {recall:.2f}')
    live.log_param('n_estimator',n_estimators)
    live.log_param('max_depth',max_depth)