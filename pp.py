import seaborn as sea
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

sea.set(style='ticks')

# Assuming you've already performed the previous preprocessing steps
# ...

# Define X and y
x = preprocessing.columns.drop('Class')
y = ['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(preprocessed[x], preprocessed[y], test_size=0.15, random_state=0)

# Train Random Forest Classifier with Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_classifier = RandomForestClassifier(random_state=0)

grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Evaluate the Model
best_rf_classifier = grid_search.best_estimator_
train_accuracy = best_rf_classifier.score(X_train, y_train)
test_accuracy = best_rf_classifier.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Generate predictions using the best Random Forest Classifier
rf_train_predictions = best_rf_classifier.predict(X_train)
rf_test_predictions = best_rf_classifier.predict(X_test)

# Combine the predictions with the original features
X_train_with_rf = pd.concat([X_train.reset_index(drop=True), pd.DataFrame({'RF_Predictions': rf_train_predictions})], axis=1)
X_test_with_rf = pd.concat([X_test.reset_index(drop=True), pd.DataFrame({'RF_Predictions': rf_test_predictions})], axis=1)

# Apply K-Means clustering on the combined data
kmeans = KMeans(n_clusters=2, random_state=0)  # Assuming 2 clusters, adjust as needed
X_train_clusters = kmeans.fit_predict(X_train_with_rf)
X_test_clusters = kmeans.predict(X_test_with_rf)

# You can now use X_train_clusters and X_test_clusters as additional features in your model

# Evaluate the performance of the combined features
# ...

# Continue with further analysis or modeling as needed
