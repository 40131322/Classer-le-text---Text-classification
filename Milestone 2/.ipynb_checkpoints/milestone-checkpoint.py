import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings

# Define stop words
stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'without', 'of', 'at', 'by', 'for', 'to',
              'in', 'on', 'from', 'up', 'down', 'out', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'don', 'should', 'now']

# Load data
X_train = np.load("D:/GitHub/Classer-le-text---Text-classification/Data/data_train.npy").astype(int)
y_train_raw = np.loadtxt('D:/GitHub/Classer-le-text---Text-classification/Data/label_train.csv', skiprows=1, delimiter=',').astype(int)
X_test = np.load("D:/GitHub/Classer-le-text---Text-classification/Data/data_test.npy").astype(int)
vocab_data = np.load("D:/GitHub/Classer-le-text---Text-classification/Data/vocab_map.npy", allow_pickle=True)

# Use the second column of y_train_raw as labels
y_raw = y_train_raw[:, 1]

# Convert vocabulary and data to DataFrame for easier manipulation with stop words
x_raw_df = pd.DataFrame(X_train, columns=vocab_data)

# Build a custom transformer for stop-word filtering
class StopWordRemover:
    def __init__(self, stop_words):
        self.stop_words = stop_words

    def transform(self, X):
        stop_word_columns = [col for col in X.columns if col in self.stop_words]
        return X.drop(columns=stop_word_columns)

    def fit(self, X, y=None):
        return self

# Suppress warnings related to constant features
warnings.filterwarnings("ignore", category=UserWarning)

# Create pipeline for feature processing, resampling, and modeling
pipeline = ImbPipeline([
    ('stop_word_remover', StopWordRemover(stop_words=stop_words)),
    ('constant_filter', VarianceThreshold()),  # Remove constant features
    ('anova_selector', SelectKBest(f_classif, k=100)),  # Select top 100 features by ANOVA F-test
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler(with_mean=False)),  # with_mean=False for sparse matrices
    ('classifier', RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
])

# Cross-validation with Stratified K-Fold
cv = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(pipeline, x_raw_df, y_raw, cv=cv, scoring='f1')

print(f"Cross-validated F1 Scores: {cv_scores}")
print(f"Mean CV F1 Score: {np.mean(cv_scores)}")

# Fit the pipeline on the full training set and predict on the test set
pipeline.fit(x_raw_df, y_raw)
X_test_df = pd.DataFrame(X_test, columns=vocab_data)
y_test_pred = pipeline.predict(X_test_df)


