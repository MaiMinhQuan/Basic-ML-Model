import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr            # learning rate
        self.n_iters = n_iters  # số lần lặp tối đa
        self.weights = None     # y = wx + b
        self.bias = None        
        
    def fit(self, X, y):        # X: training sample, y: label
        # Implement Gradient Descent
        # init parameters
        n_sample, n_feature = X.shape
        self.weights = np.zeros(n_feature)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = (1 / n_sample) * np.dot(X.T, (y_predicted - y))  # derivative respect to w
            db = (1 / n_sample) * np.sum(y_predicted - y)       # derivative respect to b
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted