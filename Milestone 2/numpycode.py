import numpy as np
import matplotlib.pyplot as plt

# Load training data
print("Loading training data...")
X_train = np.load("D:/GitHub/Classer-le-text---Text-classification/Data/data_train.npy").astype(int)
print("Training data loaded.")

print("Loading training labels...")
y_train = np.loadtxt('D:/GitHub/Classer-le-text---Text-classification/Data/label_train.csv', skiprows=1, delimiter=',').astype(int)
print("Training labels loaded.")

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

vocab_data.shape

x_raw = X_train
y_raw = y_train[:,1]
def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    class_weights = {cls: total_samples / (len(classes) * count) for cls, count in zip(classes, counts)}
    return class_weights

def weighted_binary_cross_entropy(y_true, y_pred, class_weights):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    weights = np.vectorize(class_weights.get)(y_true)
    loss = -np.mean(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    return loss
class GaussianMaxLikelihood:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.mu = np.zeros(n_dims)
        # We only save a scalar standard deviation because our model is the isotropic Gaussian
        self.sigma_sq = 1.0

    # For a training set, the function should compute the ML estimator of the mean and the variance
    def train(self, train_data,weights = None):
        # Here, you have to find the mean and variance of the train_data data and put it in self.mu and self.sigma_sq
        ### Using the expressions derived above
        if weights is None:
            weights = np.ones(train_data.shape[0])
        self.mu = np.average(train_data, axis=0, weights=weights)
        self.sigma_sq = np.average((train_data - self.mu) ** 2.0, axis=0, weights=weights).mean()


    # Returns a vector of size nb. of test ex. containing the log probabilities of each test example under the model.
    # exemple test
    def loglikelihood(self, test_data):
        ### a is the log of the first term (1/(sigma sqrt(2pi))) in the isotropic probability distribution in the image above
        ### We are using log(ab) = log(a)+log(b) where a is 1/(sigma sqrt(2pi)) and b is exp(-||x-mu||^2/(2*sigma^2)). Again see the image above.
        a = self.n_dims * -(np.log(np.sqrt(self.sigma_sq)) + (1 / 2) * np.log(2 * np.pi))
        log_prob = a - np.sum((test_data - self.mu) ** 2.0, axis=1) / (2.0 * self.sigma_sq)
        return log_prob


class BayesClassifier:
    def __init__(self, maximum_likelihood_models, priors):
        self.maximum_likelihood_models = maximum_likelihood_models ### a list of multi-variate MaxLikelihoodGaussians for each class
        self.priors = priors
        if len(self.maximum_likelihood_models) != len(self.priors):
            print('The number of ML models must be equal to the number of priors!')
        self.n_classes = len(self.maximum_likelihood_models)

    # Returns a matrix of size number of test ex. times number of classes containing the log
    # probabilities of each test example under each model, trained by ML.
    def loglikelihood(self, test_data):

        log_pred = np.zeros((test_data.shape[0], self.n_classes))

        for i in range(self.n_classes):
            # Here, we will have to use maximum_likelihood_models[i] and priors to fill in for each class
            # each column of log_pred (it's more efficient to do a entire column at a time)
            log_pred[:, i] = self.maximum_likelihood_models[i].loglikelihood(test_data) + np.log(self.priors[i])

        return log_pred
    
    # Calculate the mean of each column
column_means = np.mean(X_train, axis=0)

# Calculate the frequency of non-zero values in each column
non_zero_frequencies = np.count_nonzero(X_train, axis=0) / X_train.shape[0]

# Define the criteria
mean_threshold = 0.1
frequency_threshold = 0.1

# Filter columns based on the criteria
columns_to_keep = (column_means >= mean_threshold) & (non_zero_frequencies >= frequency_threshold)
X_train_filtered = X_train[:, columns_to_keep]

print(columns_to_keep)

def smote(X, y, minority_class, N):
    """
    Synthetic Minority Over-sampling Technique (SMOTE) implementation using numpy.
    
    Parameters:
    - X: Feature matrix
    - y: Labels
    - minority_class: The class to be oversampled
    - N: Number of synthetic samples to generate
    
    Returns:
    - X_resampled: Feature matrix after SMOTE
    - y_resampled: Labels after SMOTE
    """
    # Extract minority class samples
    X_minority = X[y == minority_class]
    
    # Number of minority samples
    n_minority_samples = X_minority.shape[0]
    
    # Number of features
    n_features = X_minority.shape[1]
    
    # Initialize synthetic samples array
    synthetic_samples = np.zeros((N, n_features))
    
    for i in range(N):
        # Randomly choose two minority samples
        idx1, idx2 = np.random.choice(n_minority_samples, 2, replace=False)
        sample1, sample2 = X_minority[idx1], X_minority[idx2]
        
        # Generate a synthetic sample
        diff = sample2 - sample1
        gap = np.random.rand()
        synthetic_sample = sample1 + gap * diff
        
        # Ensure the synthetic sample is integer
        synthetic_samples[i] = np.round(synthetic_sample).astype(int)
    
    # Combine original data with synthetic samples
    X_resampled = np.vstack((X, synthetic_samples))
    y_resampled = np.hstack((y, np.full(N, minority_class)))
    
    return X_resampled, y_resampled
minority_class = 1  # Replace with your minority class label



N = 4826  # Number of synthetic samples to generate (difference in class 0 and class 1 data)
X_resampled, y_resampled = smote(x_raw, y_raw, minority_class, N)

X_train = X_resampled
y_train = y_resampled

# Calculate the mean appearance of each word for each class
mean_appearance_class_0 = X_train[y_train == 0].mean(axis=0)
mean_appearance_class_1 = X_train[y_train == 1].mean(axis=0)

# Compute the absolute difference in mean appearance between the two classes
mean_difference = np.abs(mean_appearance_class_0 - mean_appearance_class_1)

# Identify the words with the largest differences
largest_diff_indices = np.argsort(mean_difference)[::-1]  # Sort indices in descending order of difference

top_n = 30  # Number of top words to display
top_words = [vocab_data[i] for i in largest_diff_indices[:top_n]]
print(top_words)
#Remove the useless data
# Calculate the mean of each column
column_means = X_train.mean(axis=0)

# Identify columns where the mean is zero
columns_to_drop = np.where(column_means == 0)[0]

# Drop these columns from the dataset
X_train_zero_dropped = np.delete(X_train, columns_to_drop, axis=1)

#log transform the data

x_log = np.log(X_train_zero_dropped+1)

def remove_columns_by_mean(X, threshold):
    """
    Remove columns based on the mean of the column values.
    
    Parameters:
    - X: Feature matrix
    - threshold: Mean value threshold for removing columns
    
    Returns:
    - X_filtered: Feature matrix with specified columns removed
    """
    # Calculate column mean and standard deviation for column means
    column_means = np.mean(X, axis=0)
    means = np.mean(column_means)
    stds = np.std(column_means)

    print(means)
    print(stds)
    # Identify outliers
    lower_bound = means - threshold * stds
    upper_bound = means + threshold * stds
    
    # Create a mask for rows without outliers
    mask = np.where((column_means >= lower_bound) & (column_means <= upper_bound))[0]
    

    return mask

    # Filter out rows with outliers
mask = remove_columns_by_mean(x_raw,1)   
# Define a list of stop words
stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'without', 'of', 'at', 'by', 'for', 'to', 'in', 'on', 'from', 'up', 'down', 'out', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

# Identify the indices of the stop words in the vocabulary
stop_word_indices = [i for i, word in enumerate(vocab_data) if word in stop_words]

print(stop_word_indices)
def get_mean_difference(topn,x,y):
    # Calculate the mean appearance of each word for each class
    mean_appearance_class_0 = x[y == 0].mean(axis=0)
    mean_appearance_class_1 = x[y == 1].mean(axis=0)
    
    # Compute the absolute difference in mean appearance between the two classes
    mean_difference = np.abs(mean_appearance_class_0 - mean_appearance_class_1)
  

    return mean_difference

def precision_recall(y_pred, y_true):
    """
    Calculate precision and recall using only NumPy.
    
    Parameters:
    - y_pred: Predicted labels
    - y_true: True labels
    
    Returns:
    - precision: Precision score
    - recall: Recall score
    """
    # Convert to NumPy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    # Calculate precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (FN + TP) if (FN + TP) > 0 else 0
    
    return precision, recall
def f1_score(precision,recall):
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score
def topn_get_F1_score(top_n, mean_difference, X_train, y_train):
    """
    Calculate F1 score for top N features.
    
    Parameters:
    - top_n: Number of top features to select
    - mean_difference: Array of mean differences for feature selection
    - X_train: Training feature matrix
    - y_train: Training labels
    
    Returns:
    - train_f1_score: F1 score for training data
    - val_f1_score: F1 score for validation data
    """
    # Filter the data
    top_n_indices = np.argsort(mean_difference)[-top_n:]  # Get the top n indices
    x_train_filtered = X_train[:, top_n_indices]

    import random
    random.seed(3395)
    # Randomly choose indexes for the train and val dataset, say with 80-20 split
    num_data = x_train_filtered.shape[0]
    inds = list(range(num_data))
    random.shuffle(inds)
    train_inds = inds[:int(0.8 * num_data)]
    val_inds = inds[int(0.8 * num_data):]
    
    # Split the data into train and val sets
    train_data = x_train_filtered[train_inds, :]
    train_labels = y_train[train_inds]
    val_data = x_train_filtered[val_inds, :]
    val_labels = y_train[val_inds]
        
    iris_train1 = train_data[train_labels == 0, :]
    iris_train2 = train_data[train_labels == 1, :]
    
    # We create a model per class (using maximum likelihood)
    model_class1 = GaussianMaxLikelihood(top_n)
    model_class2 = GaussianMaxLikelihood(top_n)
    
    # We train each of them using the corresponding data
    model_class1.train(iris_train1)
    model_class2.train(iris_train2)
    
    # We create a list of all our models, and the list of prior values
    total_num = len(iris_train1) + len(iris_train2)
    priors = [len(iris_train1) / total_num, len(iris_train2) / total_num]
    
    # We create our classifier with our list of Gaussian models and our priors
    classifier = BayesClassifier([model_class1, model_class2], priors)
    
    # Calculate the log-probabilities according to our model
    log_prob = classifier.loglikelihood(train_data)
    # Predict labels
    train_classes_pred = log_prob.argmax(1)
    train_precision, train_recall = precision_recall(train_classes_pred, train_labels)
    train_f1_score = f1_score(train_precision, train_recall)

    # Calculate the log-probabilities according to our model
    log_prob = classifier.loglikelihood(val_data)
    # Predict labels
    val_classes_pred = log_prob.argmax(1)
    val_precision, val_recall = precision_recall(val_classes_pred, val_labels)
    val_f1_score = f1_score(val_precision, val_recall)
    
    return train_f1_score, val_f1_score

x = X_resampled
y = y_resampled

mask = remove_columns_by_mean(x, 3)
combined_mask = [col for col in mask if col not in stop_word_indices]
mask.shape
x = np.log(1+x)

x= x[:, mask]

# Evaluate the model for different numbers of top words
max_n = 1001
top_ns = range(750, max_n,10)  # Evaluate for top 10, 20, ..., 100 words
mean_diff = get_mean_difference(max_n,x,y)
topn_get_F1_score(160,mean_diff,x,y)
train_f1_scores = []
val_f1_scores = []
for top_n in top_ns:
    train_f1, val_f1 = topn_get_F1_score(top_n,mean_diff,x,y)
    train_f1_scores.append(train_f1)
    val_f1_scores.append(val_f1)
