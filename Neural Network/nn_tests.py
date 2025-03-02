import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from nn import NeuralNetwork

# Load dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define accuracy function
def accuracy(y_true, y_predict):
    y_predict = (y_predict >= 0.5).astype(int)  # Convert probabilities to binary class labels
    return np.sum(y_true == y_predict) / len(y_true)

# Initialize and train the neural network
nn_test = NeuralNetwork([X_train.shape[1], 10, 5, 1], alpha=0.001)
nn_test.fit(X_train_scaled, y_train, epochs=2000, verbose=500)

# Predict on the test set
predictions_test = nn_test.predict(X_test_scaled)

# Compute and print accuracy
accuracy_test = accuracy(y_test, predictions_test)
print("Neural Network classification accuracy:", accuracy_test)
