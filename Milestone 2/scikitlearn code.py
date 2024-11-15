import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif, RFE
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# Load and display data
print("Loading training data...")
X_train = np.load("D:/GitHub/Classer-le-text---Text-classification/Data/data_train.npy").astype(int)
print("Training data loaded.")

# Load training labels, and immediately select only the label column (second column)
y_train_raw = np.loadtxt('D:/GitHub/Classer-le-text---Text-classification/Data/label_train.csv', skiprows=1, delimiter=',', usecols=1, dtype=int)
print("Loading training labels...")
y_train = np.loadtxt('D:/GitHub/Classer-le-text---Text-classification/Data/label_train.csv', skiprows=1, delimiter=',').astype(int)[:, 1]
print("Raw Training labels loaded.")

print("Loading test data...")
X_test = np.load("D:/GitHub/Classer-le-text---Text-classification/Data/data_test.npy").astype(int)
print("Test data loaded.")

print("Loading vocabulary...")
vocab_data = np.load("D:/GitHub/Classer-le-text---Text-classification/Data/vocab_map.npy", allow_pickle=True)
print("Vocab data loaded.")

# Check dimensions
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train_raw.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Vocab data shape: {vocab_data.shape}")

# Step 1: Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train_raw)
print(f"Resampled training data shape: {X_train_res.shape}")

# Step 2: Log transformation to stabilize variance
X_train_log = np.log1p(X_train_res)
X_test_log = np.log1p(X_test)

# Step 3: Feature selection using mutual information
mi_scores = mutual_info_classif(X_train_log, y_train_res)
top_n_features = 100  # You may adjust this number based on experimentation
top_n_indices = np.argsort(mi_scores)[-top_n_features:]
X_train_top_n = X_train_log[:, top_n_indices]
X_test_top_n = X_test_log[:, top_n_indices]

# Step 4: Recursive Feature Elimination (RFE) with XGBoost
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
n_features_to_select = 30  # Set to the desired number of final features
rfe = RFE(estimator=xgb_clf, n_features_to_select=n_features_to_select)
X_train_rfe = rfe.fit_transform(X_train_top_n, y_train_res)
X_test_rfe = rfe.transform(X_test_top_n)

# Step 5: Hyperparameter tuning with RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [1, 1.5, 2]
}

random_search = RandomizedSearchCV(
    xgb_clf, param_distributions=param_grid, n_iter=50, scoring='f1', cv=5, random_state=42, n_jobs=-1
)
random_search.fit(X_train_rfe, y_train_res)
best_model = random_search.best_estimator_

# Step 6: Cross-validation to evaluate the model
kf = StratifiedKFold(n_splits=5)
cross_val_precision, cross_val_recall, cross_val_f1 = [], [], []
conf_matrices = []

for train_index, val_index in kf.split(X_train_rfe, y_train_res):
    X_train_fold, X_val_fold = X_train_rfe[train_index], X_train_rfe[val_index]
    y_train_fold, y_val_fold = y_train_res[train_index], y_train_res[val_index]
    
    # Train the best model on the current fold
    best_model.fit(X_train_fold, y_train_fold)
    
    # Predict on the validation fold
    y_val_pred = best_model.predict(X_val_fold)
    
    # Calculate metrics
    precision = precision_score(y_val_fold, y_val_pred)
    recall = recall_score(y_val_fold, y_val_pred)
    f1 = f1_score(y_val_fold, y_val_pred)
    
    # Store metrics
    cross_val_precision.append(precision)
    cross_val_recall.append(recall)
    cross_val_f1.append(f1)
    
    # Store confusion matrix
    conf_matrices.append(confusion_matrix(y_val_fold, y_val_pred))

# Calculate and display average cross-validation scores
avg_precision = np.mean(cross_val_precision)
avg_recall = np.mean(cross_val_recall)
avg_f1 = np.mean(cross_val_f1)

print(f"Cross-Validation Precision: {avg_precision:.4f}")
print(f"Cross-Validation Recall: {avg_recall:.4f}")
print(f"Cross-Validation F1 Score: {avg_f1:.4f}")

# Plot the confusion matrix for the last fold
last_conf_matrix = conf_matrices[-1]
disp = ConfusionMatrixDisplay(confusion_matrix=last_conf_matrix, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix of the Last Validation Fold")
plt.show
