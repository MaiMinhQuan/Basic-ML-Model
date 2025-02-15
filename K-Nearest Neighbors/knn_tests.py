import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['red', 'green', '#blue'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Visualize
# print(X_train.shape)    # (120,4) -> 120 samples, 4 feature
# print(X_train[0])       # [5.1 2.5 3.  1.1]

# print(y_train.shape)
# print(y_train)

# plt.figure()
# plt.scatter(X[:, 0], X[:, 2], c=y, cmap=cmap, edgecolors='k', s=20)    # illustrate sample using feature 0, 2 (2D)
# plt.show()

from knn import KNN
clf = KNN(k=3)  #classifier
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)  # true predict / number of sample
print(accuracy)