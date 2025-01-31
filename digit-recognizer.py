import csv
import numpy as np

# Load training data
train_file = 'data/train.csv'
reader = csv.reader(open(train_file, 'r'), delimiter=',')
train = np.array(list(reader))[1:, ].astype('float')
Y_train = train[:, 0]
X_train = train[:, 1:]
N_train = X_train.shape[0]
D = X_train.shape[1]
K = int(np.max(Y_train)) + 1
T_train = np.zeros((N_train, K))

for i in range(N_train):
    T_train[i, int(Y_train[i])] = 1

# Add a column of ones to X_train for the bias term
X_train = np.hstack([np.ones((N_train, 1)), X_train])

W = np.random.randn(D + 1, K)  # Adjust the shape of W to include the bias term

print(N_train, D, len(Y_train))

def softmax(a):
    c = np.max(a, axis=1, keepdims=True)
    expA = np.exp(a - c)
    return expA / expA.sum(axis=1, keepdims=True)

def predict(X, W):
    return softmax(X.dot(W))

def cost(T, Y):
    return -(T * np.log(Y)).sum()

def grad(Y, T, X):
    return X.T.dot(Y - T)

def classify(Y, T, X, W, epochs, learning_rate):
    for i in range(epochs):
        Y = predict(X, W)
        W -= learning_rate * grad(Y, T, X)
    print(W)

# Load test data
test_file = 'data/test.csv'
reader = csv.reader(open(test_file, 'r'), delimiter=',')
test = np.array(list(reader))[1:, ].astype('float')
X_test = test[:, 1:]
Y_test = test[:, 0]

# Add a column of ones to X_test for the bias term
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Train the model
if __name__ == '__main__':
    classify(Y_train, T_train, X_train, W, int(N_train/1), 0.01)

    # Make predictions on the test data
    Y_pred = predict(X_test, W)
    Y_pred_labels = np.argmax(Y_pred, axis=1)

    # Calculate accuracy
    accuracy = np.mean(Y_pred_labels == Y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')