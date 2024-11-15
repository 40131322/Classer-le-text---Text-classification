import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix

# Load and display data shape
X_train = np.load("D:/GitHub/Classer-le-text---Text-classification/Data/data_train.npy").astype(int)
y_train = np.loadtxt('D:/GitHub/Classer-le-text---Text-classification/Data/label_train.csv', skiprows=1, delimiter=',').astype(int)[:, 1]
X_test_raw = np.load("D:/GitHub/Classer-le-text---Text-classification/Data/data_test.npy").astype(int)
vocab_data_raw = np.load("D:/GitHub/Classer-le-text---Text-classification/Data/vocab_map.npy", allow_pickle=True)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test_raw.shape}")
print(f"Vocabulary data shape: {vocab_data_raw.shape}")
# Load data



# Convert to sparse format
X_train_sparse = csr_matrix(X_train)

# Split into train and validation sets
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train_sparse, y_train, test_size=0.2, random_state=42)

# Preprocessing (TF-IDF)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_sub)
X_val_tfidf = tfidf_transformer.transform(X_val)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train_sub)

# Predict and calculate F1 score on validation set
y_val_pred = model.predict(X_val_tfidf)
f1_val = f1_score(y_val, y_val_pred, average='macro')
print("Validation Macro F1 Score:", f1_val)
