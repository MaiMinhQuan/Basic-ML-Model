import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
    
class KNN:
    def __init__(self, k = 3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
      
    def _predict(self, x):  # predict 1 sample
        # compute distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]  # return an array of sorted index
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1) # 1 most common item -> return a list: each item (value, frequency)
        return most_common[0][0] # label
        
    def predict(self, X):   # predict samples
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
