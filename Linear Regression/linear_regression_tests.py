import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234) # Split dataset, train: 80%, test: 20%

# Visualize sample dataset
# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

# print(X_train.shape)    # (80, 1) -> 80 samples, 1 feature 
# print(y_train.shape)    # (80,) -> 80 labels

from linear_regression import LinearRegression

regressor = LinearRegression(lr=0.01)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

def mse(y_true, y_predicted):   # mean square error (loss function)
    return np.mean((y_true - y_predicted) ** 2)

mse_value = mse(y_test, predicted)
print(mse_value)

y_predict_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color="blue", s=10)
m2 = plt.scatter(X_test, y_test, color="red", s=10)
plt.plot(X, y_predict_line, color="black", linewidth=2, label="Prediction")
plt.show()